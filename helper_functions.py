import shutil
import torch
import os
import zipfile
from pathlib import Path
from tqdm.auto import tqdm
import requests


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

def one_hot_encode(y, num_classes):
    return torch.eye(num_classes)[y]

# Training step
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device,
               ep: int,
               EPOCHS: int):
    """ Perform a single training step for the model."""

    # Set the model to train mode
    model.train()

    # Initialize the loss and accuracy
    train_loss, train_acc = 0, 0
    all_preds = []
    all_targets = []
    # Wrap the data loader with tqdm
    with tqdm(total=len(data_loader), desc=f'[Epoch {ep+1}/{EPOCHS}] Training') as pbar:
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

            # Accumulate predictions and targets
            all_preds.extend(y_pred_class.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
                
            # Update tqdm progress bar
            pbar.postfix = f"Loss: {train_loss/(batch+1):.4f} | Accuracy: {(train_acc/(batch+1))*100:.4f}%"
            pbar.update(1)

    # Calculate average loss and accuracy
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    return train_loss, train_acc, all_preds, all_targets


# Testing step
import numpy as np  # Add this line to import NumPy
def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device,
              ep: int,
              EPOCHS: int):
    """ Perform a single testing step for the model."""

    # Set the model to evaluation mode
    model.eval()

    # Initialize test loss and accuracy
    test_loss, test_acc = 0, 0
    all_preds = []
    all_targets = []
    
    with tqdm(total=len(data_loader), desc=f'[Epoch {ep+1}/{EPOCHS}] Testing') as pbar:
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

                # Append predicted probabilities and targets
                all_preds.extend(torch.softmax(test_pred_logits, dim=1).cpu().numpy())
                all_targets.extend(y.cpu().numpy())

                # Update tqdm progress bar
                pbar.postfix = f"Loss: {test_loss/(batch+1):.4f} | Accuracy: {(test_acc/(batch+1))*100:.4f}%"
                pbar.update(1)

        # Adjust metrics to get the average loss and accuracy
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)


    return test_loss, test_acc, np.array(all_preds), np.array(all_targets)



# Training and testing the model
def train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          test_loader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device,
          scheduler: torch.optim.lr_scheduler = None,
          early_stopping: bool = True,
          times_not_optimizing: int = 1,
          acc_target: float = 0.95):
    """ Train the model and evaluate on the test set."""

    results = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'all_preds': [],
        'all_targets': [],
        'train_all_preds': [[]],
        'train_all_targets': [[]],
        'test_all_preds': [[]],
        'test_all_targets': [[]]
    }

    count = 0
    best_model_weights = model.state_dict()
    best_test_loss = float('inf')
    wanted_acc = False

    for epoch in range(epochs):
        train_loss, train_acc, train_all_preds, train_all_targets = train_step(model=model,
                                                                                data_loader=train_loader,
                                                                                loss_fn=loss_fn,
                                                                                optimizer=optimizer,
                                                                                device=device,
                                                                                ep=epoch,
                                                                                EPOCHS=epochs)

        test_loss, test_acc, test_all_preds, test_all_targets = test_step(model=model,
                                                                            data_loader=test_loader,
                                                                            loss_fn=loss_fn,
                                                                            device=device,
                                                                            ep=epoch,
                                                                            EPOCHS=epochs)
        
        if train_acc > acc_target:
            wanted_acc = True
            print("Reached Wanted Accuracy")
        else:
            wanted_acc = False


        if early_stopping and wanted_acc:
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model_weights = model.state_dict()
                print(f"Saving Best weights and reseting the counter to 0 from {count}")
                count = 0  # Reset count when there is an improvement
            else:
                count += 1  # Increment count if no improvement

            if count >= times_not_optimizing:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_weights)
                break

        # Save the results
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)
        results['train_all_preds'].append(train_all_preds)
        results['train_all_targets'].append(train_all_targets)
        results['test_all_preds'].append(test_all_preds)
        results['test_all_targets'].append(test_all_targets)

    if scheduler:
        scheduler.step()

    return results



# Save the model
def save_model(module, acc=None, hyperparameters=None, total_time=None, fold=None):
    # Create the base directory if it doesn't exist
    base_dir = "models"
    model_name = module.get_name()
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Create the sub folder for the model if it doesn't exist
    model_dir = os.path.join(base_dir, f"{model_name}_accuracy[{acc*100:.2f}%]_{hyperparameters['DEVICE']}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if hyperparameters:
        # Create a file that contains info about the model and the hyperparameters
        with open(os.path.join(model_dir, "info.txt"), "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Accuracy: {acc*100:.2f}%\n")
            f.write("Hyperparameters:\n")
            f.write(f"Total time: {total_time} hh:mm:ss\n")
            f.write(f"Fold combination: {fold}\n")
            for key, value in hyperparameters.items():
                f.write(f"{key}: {value}\n")

    # Save the model
    model_path = os.path.join(model_dir, f"{model_name}.pt")
    torch.save(module.state_dict(), model_path)
    print(f"Model saved at {model_path}")
    return model_dir


from typing import Tuple, Dict, List

# Find classes in a directory
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx


import os
from PIL import Image

def convert_png_to_jpg(png_path, jpg_path):
    img = Image.open(png_path)
    rgb_img = img.convert('RGB')
    rgb_img.save(jpg_path, 'JPEG')

def combine(classes, source, dest):
    # Create the destination folder if it doesn't exist
    os.makedirs(dest, exist_ok=True)
    
    # Initialize a counter for the image number
    image_number = 1
    
    for class_name, class_code in classes.items():
        # Convert class_code to string
        class_code_str = str(class_code)
        
        # Get the list of files for this class
        file_list = os.listdir(os.path.join(source, class_name))
        
        # Iterate over the files in this class
        for file_name in file_list:
            # Construct the full file path
            file_path = os.path.join(source, class_name, file_name)
            
            # Construct the new file name
            new_file_name = f"{class_code_str}_{image_number}.jpg"
            image_number += 1
            
            # Construct the destination file path
            dest_file_path = os.path.join(dest, new_file_name)
            
            # If the file is a PNG, convert it to JPG
            if file_path.lower().endswith('.png'):
                # Convert the PNG to JPG
                convert_png_to_jpg(file_path, dest_file_path)
            else:
                # Otherwise, just move the file
                shutil.move(file_path, dest_file_path)

from sklearn.metrics import roc_curve, roc_auc_score, recall_score
import matplotlib.pyplot as plt

def calculate_metrics(all_preds, all_targets, num_classes):
    # Calculate ROC and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    sensitivities = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_targets == i, all_preds[:, i])
        roc_auc[i] = roc_auc_score(all_targets == i, all_preds[:, i])
        sensitivities[i] = recall_score(all_targets, np.argmax(all_preds, axis=1), average=None)[i]

    return fpr, tpr, roc_auc, sensitivities



# Evaluate the model
def evaluate(model: torch.nn.Module,
             data_loader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device,
             eval_epochs: int = 1):
    """ Evaluate the model on the test set."""

    # Evaluation dict
    eval_results = {
        'test_loss': 0,
        'test_acc': 0,
        'all_preds': [],
        'all_targets': [],
        'roc_auc': {},
        'sensitivity': {}
    }
    
    for i in range(eval_epochs):
        test_loss, test_acc, all_preds, all_targets = test_step(model=model,
                                                                data_loader=data_loader,
                                                                loss_fn=loss_fn,
                                                                device=device,
                                                                ep=i,
                                                                EPOCHS=eval_epochs)

        eval_results['test_loss'] += test_loss
        eval_results['test_acc'] += test_acc
        eval_results['all_preds'].extend(all_preds)
        eval_results['all_targets'].extend(all_targets)

    # Average the results
    eval_results['test_loss'] /= eval_epochs
    eval_results['test_acc'] /= eval_epochs

    # Calculate ROC, AUC, and sensitivity
    fpr, tpr, roc_auc, sensitivities = calculate_metrics(np.array(eval_results['all_preds']),
                                                         np.array(eval_results['all_targets']),
                                                         num_classes=4)

    eval_results['roc_auc'] = roc_auc
    eval_results['sensitivity'] = sensitivities

    # Print the results
    print("Evaluation results")
    print(f"Test Loss: {eval_results['test_loss']:.4f} | Test Acc: {eval_results['test_acc']*100:.2f}%")
    for i in range(4):
        print(f'Class {i} - AUC: {roc_auc[i]:.2f}, Sensitivity: {sensitivities[i]:.2f}')

    return eval_results


import pandas as pd
import pathlib
from tqdm import tqdm

def load_data_to_dataframe(dir):
    # Load the image paths to numpy array
    image_paths = list(pathlib.Path(dir).glob('*.jpg'))

    # Create a dataframe to store image paths and labels
    data = {'image_path': [], 'label': []}

    with tqdm(total=len(image_paths), desc="Loading images") as pbar:
        for image_path in image_paths:
            class_idx = int(image_path.name.split('_')[0])  # Extract label from file name
            data['image_path'].append(str(image_path))  # Store path as string
            data['label'].append(class_idx)
            pbar.update(1)

    # Create a pandas dataframe from the data
    df = pd.DataFrame(data)
    return df

def combine_and_rename_images(data_dir, classes):
    # Create a combined directory for both training and testing sets

    combined_dir = 'Combined'
    os.makedirs(combined_dir, exist_ok=True)

    count_classes = classes

    count_classes = {key : 0 for key in count_classes}


    # Iterate through training and testing directories
    with tqdm(total=2, desc="Combining and renaming images") as pbar:

        for subset in ['Training', 'Testing']:
            subset_path = os.path.join(data_dir, subset)
            for class_dir in os.listdir(subset_path):
                class_path = os.path.join(subset_path, class_dir)
                for image in os.listdir(class_path):
                    # Get the class of the original image
                    old_image_path = os.path.join(class_path, image)

                    # Get the new name of the image
                    new_image_name = f"{classes[class_dir]}_{count_classes[class_dir]}.jpg"
                    count_classes[class_dir] += 1

                    # Get the path of the new image
                    new_image_path = os.path.join(combined_dir, new_image_name)

                    # Copy the image to the combined directory
                    shutil.copy(old_image_path, new_image_path)

            pbar.update(1)
    print("Images combined and renamed successfully.")
    return combined_dir