# MRI Image Classification with Convolutional Neural Networks (CNN-Classification)

CNN-Classification is a Python project designed for MRI image classification tasks. It includes a custom dataset class for loading MRI images and a convolutional neural network (CNN) model designed for classification. The project provides helper functions for data processing, model training, evaluation, and visualization, along with Jupyter notebooks for experimentation.

## Overview

- **classes.py**: Defines a `CustomDataset` class to load MRI images and their corresponding class labels. It utilizes PyTorch's `Dataset` class and `torchvision`'s transforms for preprocessing. The `MRI_Classification_CNN` model is custom-built using PyTorch's `nn.Module` for MRI image classification tasks.
- **helper_functions.py**: Contains various helper functions for data processing, training, evaluation, and saving the model.
- **plot_functions.py**: Includes functions for visualizing performance metrics, such as loss curves and accuracy across classes.
- **notebooks/**: Contains Jupyter notebooks that demonstrate training, evaluation, and testing of the CNN model. These notebooks serve as tutorials and experiments, allowing users to modify the pipeline and test the performance of different models.

## File Structure

- **classes.py**: Contains the implementation of `CustomDataset` and `MRI_Classification_CNN`.
- **helper_functions.py**: Contains utility functions for loading data, training the model, evaluating performance, and saving results.
- **plot_functions.py**: Contains plotting functions to visualize training metrics, such as loss curves and accuracy.
- **notebooks/**: Jupyter notebooks showcasing the training and evaluation of the CNN model, as well as experiments with different configurations.

## Dataset

The dataset used is the [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) dataset from Kaggle. It includes MRI images of brain tumors along with their class labels.

## Classes

The key classes used in this project are:

- **CustomDataset**: A PyTorch `Dataset` class for loading MRI images and their corresponding labels. It applies necessary transformations like resizing, normalization, and augmentations.
- **MRI_Classification_CNN**: The custom CNN model designed specifically for MRI image classification. It includes multiple convolutional layers, pooling layers, and fully connected layers to classify MRI images into their respective categories.

## Model Architecture

The custom CNN model consists of several layers of convolutions, batch normalization, activation functions, and pooling. The architecture is tuned to effectively classify MRI brain images into their respective categories.

## Dependencies

- Python (>= 3.9)
- PyTorch (>= 2.0)
- torchvision
- Pillow (PIL)
- Matplotlib
- NumPy
- tqdm
- Jupyter Notebook

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/theodorostaloumtzis/MRI-Classification.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the dataset:
   - Download the [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) dataset from Kaggle.
   - Extract the dataset into a `data/` directory.


## Contributors

- [Theodoros Taloumtzis](https://github.com/theodorostaloumtzis)
