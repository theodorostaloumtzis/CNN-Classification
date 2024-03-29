from torch.utils.data import Dataset
from PIL import Image
from helper_functions import *
from math import ceil
import pathlib

basic_mb_params = [
    # k, channels(c), repeats(t), stride(s), kernel_size(k)
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

alpha, beta = 1.2, 1.1

scale_values = {
    # (phi, resolution, dropout)
    "b0": (0, 224, 0.2),
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}


# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self,
                 targ_dir: str,
                 transform=None):
        self.paths = list(pathlib.Path(targ_dir).glob('*/*.jpg'))
        # Setup transform
        self.transform = transform
        # Create class to index mapping
        self.classes, self.class_to_idx = find_classes(targ_dir)

    def load_image(self, index: str) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Returns the image and class label at the given index"""
        image = self.load_image(index)
        class_name = self.paths[index].parent.name  # extract class name from parent folder data/class_name/image.jpg
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(image), class_idx  # return transformed image and class index (X, y)
        else:
            return image, class_idx  # return image and class index (X, y) untransformed


class MRI_classification_CNN(nn.Module):

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, size: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units * 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units * 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_units * 3, hidden_units * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_units * 2, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_units * size // 4 * size // 4, hidden_units * 4),
            nn.ReLU(),
            nn.Linear(hidden_units * 4, hidden_units * 2),
            nn.ReLU(),
            nn.Linear(hidden_units * 2, output_shape)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, groups=1):
        super(ConvBlock, self).__init__()
        self.cnnblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.SiLU())

    def forward(self, x):
        return self.cnnblock(x)


class MBBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, expand_ratio, reduction=2):
        super(MBBlock, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim

        reduced_dim = max(1, int(in_channels / reduction))

        if self.expand:
            self.expand_conv = ConvBlock(in_channels, hidden_dim,
                                         kernel_size=1, stride=1, padding=0)

        self.conv = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim, kernel_size,
                      stride, padding, groups=hidden_dim),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, inputs):
        if self.expand:
            x = self.expand_conv(inputs)
        else:
            x = inputs
        return self.conv(x)



class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class EfficientNet(nn.Module):
    def __init__(self, model_name, output):
        super(EfficientNet, self).__init__()
        phi, resolution, dropout = scale_values[model_name]
        self.depth_factor, self.width_factor = alpha ** phi, beta ** phi
        self.last_channels = ceil(1280 * self.width_factor)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_extractor()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.last_channels, output),
        )

    def feature_extractor(self):
        channels = int(32 * self.width_factor)
        features = [ConvBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels

        for k, c_o, repeat, s, n in basic_mb_params:
            # For numeric stability, we multiply and divide by 4
            out_channels = 4 * ceil(int(c_o * self.width_factor) / 4)
            num_layers = ceil(repeat * self.depth_factor)

            for layer in range(num_layers):
                if layer == 0:
                    stride = s
                else:
                    stride = 1
                features.append(
                    MBBlock(in_channels, out_channels, expand_ratio=k,
                            stride=stride, kernel_size=n, padding=n // 2)
                )
                in_channels = out_channels

        features.append(
            ConvBlock(in_channels, self.last_channels,
                      kernel_size=1, stride=1, padding=0)
        )
        self.extractor = nn.Sequential(*features)

    def forward(self, x):
        x = self.avgpool(self.extractor(x))
        return self.classifier(self.flatten(x))