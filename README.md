# CNN-Classification: MRI Image Classification with Convolutional Neural Networks

CNN-Classification is a Python project designed for MRI image classification tasks. It includes a custom dataset class for loading MRI images and a convolutional neural network (CNN) model for classification. This README provides an overview of the project structure, usage instructions, and other relevant details.

## Overview

- **classes.py**: Defines a `CustomDataset` class to load MRI images and their corresponding class labels. It utilizes PyTorch's Dataset class and torchvision's transforms for preprocessing. Also, contains the `MRI_classification_CNN` model built using PyTorch's nn.Module for classification tasks.
- **helper_functions.py**: Contains various helper functions for data processing, model training, evaluation, and visualization.
- **cnn.py**: Orchestrates the training and evaluation process. It imports necessary classes and functions, defines hyperparameters, creates datasets and data loaders, trains the CNN model, and evaluates its performance. Additionally, saves the trained model, plots loss curves, data distribution, and accuracy per class.

## File Structure

- **classes.py**: Contains the implementation of `CustomDataset` and `MRI_classification_CNN`.
- **helper_functions.py**: Contains various helper functions.
- **cnn.py**: Orchestrates training and evaluation.

## Dataset

The dataset used is the [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) dataset from Kaggle. It includes MRI images of brain tumors along with their class labels.

## Dependencies

- Python (>= 3.9)
- PyTorch (>= 2.0)
- torchvision
- Pillow (PIL)
- Matplotlib
- tqdm

## Usage

1. **Dataset Creation**:
   - Instantiate `CustomDataset` by providing the target directory containing MRI images. Optionally, apply transformations for preprocessing.

2. **Model Definition and Training**:
   - Instantiate `MRI_classification_CNN`, specifying input shape, hidden units, output shape, and image size. Train the model using appropriate training data.

3. **Inference**:
   - Use the trained model for inference on new MRI images.

4. **Execution (cnn.py)**:
   - Run `cnn.py` to train and evaluate the CNN model. Ensure correct data directory paths (`TRAIN_DIR` and `TEST_DIR`). Adjust hyperparameters as needed.

## Contributors

- [Theodoros Taloumtzis](https://github.com/theodorostaloumtzis)

