"""
A series of helper functions used throughout the course.

If a function gets defined once and could be used over and over, it'll go in here.
"""
import pathlib

import torch
import matplotlib.pyplot as plt
import numpy as np

from torch import nn

import os
import zipfile

from pathlib import Path
from tqdm.auto import tqdm

import requests

# Walk through an image classification directory and find out how many files (images)
# are in each subdirectory.
import os


def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
    dir_path (str): target directory

    Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


# Plot linear data or training and test and predictions (optional)
def plot_predictions(
        train_data, train_labels, test_data, test_labels, predictions=None
):
    """
  Plots linear training data and test data and compares predictions.
  """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


# Plot loss curves of a model
def plot_loss_curves(results, model_dir=None):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid(True)
    if model_dir:
        plt.savefig(os.path.join(model_dir, "loss.png"))

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid(True)
    if model_dir:
        plt.savefig(os.path.join(model_dir, "accuracy.png"))

    plt.show()


# Pred and plot image function from notebook 04 See creation:
# https://www.learnpytorch.io/04_pytorch_custom_datasets/#113-putting-custom-image-prediction-together-building-a-function
from typing import List
import torchvision


def pred_and_plot_image(
        model: torch.nn.Module,
        image_path: str,
        class_names: List[str] = None,
        transform=None,
        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Makes a prediction on a target image with a trained model and plots the image.

    Args:
        model (torch.nn.Module): trained PyTorch image classification model.
        image_path (str): filepath to target image.
        class_names (List[str], optional): different class names for target image. Defaults to None.
        transform (_type_, optional): transform of target image. Defaults to None.
        device (torch.device, optional): target device to compute on. Defaults to "cuda" if torch.cuda.is_available() else "cpu".
    
    Returns:
        Matplotlib plot of target image and model prediction as title.

    Example usage:
        pred_and_plot_image(model=model,
                            image="some_image.jpeg",
                            class_names=["class_1", "class_2", "class_3"],
                            transform=torchvision.transforms.ToTensor(),
                            device=device)
    """

    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.0

    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )  # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)


def set_seeds(seed: int = 42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


def download_data(source: str,
                  destination: str,
                  remove_source: bool = True) -> Path:
    """Downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.
    
    Returns:
        pathlib.Path to downloaded data.
    
    Example usage:
        download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                      destination="pizza_steak_sushi")
    """
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

        # Download pizza, steak, sushi data
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...")
            zip_ref.extractall(image_path)

        # Remove .zip file
        if remove_source:
            os.remove(data_path / target_file)

    return image_path


# Calculate  class accuracy


# Training step
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device):
    """ Perform a single training step for the model."""

    # Set the model to train mode
    model.train()

    # Initialize the loss and accuracy
    train_loss, train_acc = 0, 0
    train_class_acc = {
        0: 0,
        1: 0,
        2: 0,
        3: 0
    }

    # Wrap the data loader with tqdm
    with tqdm(total=len(data_loader), desc='Training') as pbar:
        # Loop over the training batches
        for batch, (X, y) in enumerate(data_loader):
            # Put data on target device
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_pred = model(X)

            # Compute loss and accuracy per batch
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()  # Accumulate the loss

            # Optimization zero_grad
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            # Accumulate accuracy
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).float().mean().item()  # formula for accuracy = correct/total

            # Accumulate class accuracy
            for i in range(len(y)):
                train_class_acc[y[i].item()] += (y_pred_class[i] == y[i]).float().item()

            # Update tqdm progress bar
            pbar.update(1)

    # Calculate average loss and accuracy
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    train_class_acc = {k: v / len(data_loader) for k, v in train_class_acc.items()}

    return train_loss, train_acc, train_class_acc


# Testing step
def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device):
    """ Perform a single testing step for the model."""

    # Set the model to evaluation mode
    model.eval()

    # Initialize test loss and accuracy
    test_loss, test_acc = 0, 0
    test_class_acc = {
        0: 0,
        1: 0,
        2: 0,
        3: 0
    }

    # Loop over the testing batches
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            # Put data on target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Compute loss per batch
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()  # Accumulate the loss

            # Accumulate accuracy
            test_pred_labels = torch.argmax(torch.softmax(test_pred_logits, dim=1), dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

            # Accumulate class accuracy
            for i in range(len(y)):
                test_class_acc[y[i].item()] += (test_pred_labels[i] == y[i]).float().item()

        # Adjust metrics to get the average loss and accuracy
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

        test_class_acc = {k: v / len(data_loader) for k, v in test_class_acc.items()}

    return test_loss, test_acc, test_class_acc


# Training and testing the model
def train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          test_loader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device,
          scheduler: torch.optim.lr_scheduler = None):
    """ Train the model and evaluate on the test set."""

    # Track the losses and accuracies
    results = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'train_acc_per_class': {0: [],
                                1: [],
                                2: [],
                                3: []},
        'test_acc_per_class': {0: [],
                               1: [],
                               2: [],
                               3: []}

    }

    # Train the model
    for epoch in range(epochs):
        train_loss, train_acc, train_class_acc = train_step(model=model,
                                                            data_loader=train_loader,
                                                            loss_fn=loss_fn,
                                                            optimizer=optimizer,
                                                            device=device)

        # Evaluate the model on the test set
        test_loss, test_acc, test_class_acc = test_step(model=model,
                                                        data_loader=test_loader,
                                                        loss_fn=loss_fn,
                                                        device=device)

        # Print the metrics
        print(
            f"Epoch: {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        # Save the results
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)

        for i in range(4):
            results['train_acc_per_class'][i].append(train_class_acc[i])
            results['test_acc_per_class'][i].append(test_class_acc[i])

    # Step the scheduler
    if scheduler:
        scheduler.step()

    return results


def save_model(module, model_name, acc=None, hyperparameters=None):
    # Create the base directory if it doesn't exist
    base_dir = "models"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Create the sub folder for the model if it doesn't exist
    model_dir = os.path.join(base_dir, f"{model_name}_accuracy[{acc*100:.2f}%]")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if hyperparameters:
        # Create a file that contains info about the model and the hyperparameters
        with open(os.path.join(model_dir, "info.txt"), "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Accuracy: {acc*100:.2f}%\n")
            f.write("Hyperparameters:\n")
            for key, value in hyperparameters.items():
                f.write(f"{key}: {value}\n")

    # Save the model
    model_path = os.path.join(model_dir, f"{model_name}.pt")
    torch.save(module.state_dict(), model_path)
    print(f"Model saved at {model_path}")
    return model_dir


def plot_category_distribution(train_dataset, test_dataset, model_dir=None):
    # Get the class names/categories from the dataset
    classes = train_dataset.classes
    print("Classes: ", classes)

    # Count occurrences of each category in the training dataset
    train_counts = [0] * len(classes)
    for _, label in train_dataset:
        train_counts[label] += 1

    # Count occurrences of each category in the test dataset
    test_counts = [0] * len(classes)
    for _, label in test_dataset:
        test_counts[label] += 1

    # Plot the bar graph
    width = 0.35
    x = np.arange(len(classes))

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, train_counts, width, label='Train Dataset')
    rects2 = ax.bar(x + width / 2, test_counts, width, label='Test Dataset')

    # x-axis labels.
    ax.set_xlabel('Categories')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Category Distribution in Train and test Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.legend()

    if model_dir:
        plt.savefig(os.path.join(model_dir, "category_distribution.png"))

    plt.show()


from typing import Tuple, Dict, List


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx


import matplotlib.pyplot as plt


def plot_accuracy_per_class(results, classes=None, model_dir=None):
    train_acc_per_class = results.get('train_acc_per_class', {})
    test_acc_per_class = results.get('test_acc_per_class', {})

    if not train_acc_per_class or not test_acc_per_class:
        print("Accuracy per class data is not available.")
        return

    num_classes = len(train_acc_per_class)
    epochs = range(1, len(train_acc_per_class[0]) + 1)

    fig, ax = plt.subplots(num_classes, 1, figsize=(10, 5 * num_classes), sharex=True)
    if num_classes == 1:
        ax = [ax]  # Ensure ax is iterable

    for i in range(num_classes):
        ax[i].plot(epochs, train_acc_per_class[i], label=f'Train Class {classes[i]} Accuracy')
        ax[i].plot(epochs, test_acc_per_class[i], label=f'Test Class {classes[i]} Accuracy')
        ax[i].set_ylabel('Accuracy')
        ax[i].set_title(f'Class {i} Accuracy')
        ax[i].legend()
        ax[i].grid(True)


    plt.xlabel('Epochs')
    plt.tight_layout()
    if model_dir:
        plt.savefig(os.path.join(model_dir, f"class_accuracy.png"))
    plt.show()





