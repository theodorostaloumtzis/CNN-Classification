from torch.utils.data import Dataset
from math import ceil
import torch.nn as nn
import torch
#from torchinfo import summary

from collections import OrderedDict


# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data =  data # List of data
        self.labels = torch.tensor(labels, dtype=torch.long) # List of labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]



class MRI_classification_CNN(nn.Module):

    def __init__(self, input_channels: int, hidden_units: int, output_shape: int, size: int, dropout=None):
        super().__init__()
        self.name = "MRI_classification_CNN"
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_units * 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units * 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_units * 3, hidden_units * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_units * 2, hidden_units * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_units * 2, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Calculate the output size after convolution layers
        self.fc_input_size = hidden_units * (size // 4) * (size // 4)  # Assuming two maxpool layers

        self.fc1 = nn.Sequential(
            nn.Linear(self.fc_input_size, 192),
            nn.ReLU(),
            nn.Linear(192, output_shape)
        )
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, x):
        # Expected input shape: (batch_size, input_channels, height, width)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc1(x)
        return x

    def get_name(self):
        return self.name
