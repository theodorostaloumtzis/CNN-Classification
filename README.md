# CNN-Classification: Custom Dataset and CNN for MRI Image Classification

This project consists of Python code defining a custom dataset class for loading MRI images and a convolutional neural network (CNN) for classification tasks.

## Overview

- **classes.py**: This file contains two main components:
  - `CustomDataset`: A custom PyTorch dataset class for loading MRI images and their corresponding class labels. It utilizes PyTorch's Dataset class and torchvision's transforms for preprocessing.
  - `MRI_classification_CNN`: A CNN model built using PyTorch's nn.Module class for MRI image classification tasks.

- **helper_functions.py**: Contains various helper functions used for data processing, model training, evaluation, and visualization.

- **cnn.py**: This script orchestrates the training and evaluation process. It imports the necessary classes and functions from `helper_functions.py` and `classes.py`, defines hyperparameters, creates datasets and data loaders, trains the CNN model, and evaluates its performance. Additionally, it saves the trained model, plots loss curves, distribution of data, and accuracy per class.

## File Structure

- **classes.py**: Contains the implementation of the `CustomDataset` class and the `MRI_classification_CNN` model.
- **helper_functions.py**: Contains various helper functions used for data processing, model training, evaluation, and visualization.
- **cnn.py**: Orchestrates the training and evaluation process.

## Dataset

The dataset used for this project is the [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) dataset acquired from Kaggle. This dataset contains MRI images of brain tumors along with their corresponding class labels.

## Dependencies

- Python (>= 3.6)
- PyTorch (>= 1.0)
- torchvision
- Pillow (PIL)
- Matplotlib
- tqdm

## Usage

1. **Dataset Creation**:
   - Instantiate the `CustomDataset` class by providing the target directory containing the MRI images.
   - Optionally, you can provide transformations for data augmentation or preprocessing.

2. **Model Definition and Training**:
   - Instantiate the `MRI_classification_CNN` model, specifying input shape, hidden units, output shape, and image size.
   - Train the model using appropriate training data.

3. **Inference**:
   - After training, use the trained model for inference on new MRI images.

4. **Execution (cnn.py)**:
   - Run the `cnn.py` script to train and evaluate the CNN model. Ensure that the data directories (`TRAIN_DIR` and `TEST_DIR`) are correctly set.
   - Adjust hyperparameters as needed.

## Contributors

- [Taloumtzis Theododoros](https://github.com/theodorostaloumtzis)
