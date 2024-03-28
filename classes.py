import pathlib
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from typing import Tuple
from torch import nn
from helper_functions import find_classes


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
            nn.Conv2d(input_shape, hidden_units * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_units * 2, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_units * size // 4 * size // 4, hidden_units * 4),
            nn.ReLU(),
            nn.Linear(hidden_units * 4, hidden_units * 2),
            nn.ReLU(),
            nn.Linear(hidden_units * 2, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_shape)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
