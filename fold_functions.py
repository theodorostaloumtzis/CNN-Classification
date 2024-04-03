import datetime
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import os
import zipfile
from pathlib import Path
from tqdm.auto import tqdm
import requests
import seaborn as sns
from sklearn.metrics import confusion_matrix
import shutil
from torch.utils.data import ConcatDataset


def combine_and_rename_images(data_dir):
    # Create a combined directory for both training and testing sets
    combined_dir = os.path.join(data_dir, 'Combined')
    os.makedirs(combined_dir, exist_ok=True)
    
    # Iterate through training and testing directories
    for subset in ['Training', 'Testing']:
        subset_dir = os.path.join(data_dir, subset)
        for class_dir in os.listdir(subset_dir):
            class_path = os.path.join(subset_dir, class_dir)
            if os.path.isdir(class_path):
                # Create a subdirectory for the class within the combined directory
                combined_class_dir = os.path.join(combined_dir, class_dir)
                os.makedirs(combined_class_dir, exist_ok=True)
                
                # Iterate through images in each class directory
                image_files = os.listdir(class_path)
                for i, image_file in enumerate(image_files):
                    old_image_path = os.path.join(class_path, image_file)
                    # Rename the image file
                    new_image_name = f"{class_dir}_{i+1}.jpg"  # Assuming images are in jpg format
                    new_image_path = os.path.join(combined_class_dir, new_image_name)
                    shutil.copyfile(old_image_path, new_image_path)
    
    print("Images combined and renamed successfully.")
    return combined_dir


def create_folds(data_dir: str, output_dir: str, k_folds: int):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of class directories
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    class_dirs.sort()

    # Calculate number of images per class per fold
    num_images_per_class = {}
    for class_dir in class_dirs:
        num_images = len(os.listdir(os.path.join(data_dir, class_dir)))
        num_images_per_fold = num_images // k_folds
        num_images_per_class[class_dir] = num_images_per_fold

    # Create the number of the folders for each fold
    for fold in range(k_folds):
        fold_dir = os.path.join(output_dir, f"Fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)

        # Create a subdirectory for each class in the fold
        for class_dir in class_dirs:
            class_dir_path = os.path.join(fold_dir, class_dir)
            os.makedirs(class_dir_path, exist_ok=True)

            # Copy the images to the class directory
            num_images_per_fold = num_images_per_class[class_dir]
            start_index = fold * num_images_per_fold
            end_index = start_index + num_images_per_fold
            image_files = os.listdir(os.path.join(data_dir, class_dir))
            for image_file in image_files[start_index:end_index]:
                image_path = os.path.join(data_dir, class_dir, image_file)
                shutil.copy(image_path, class_dir_path)


    print(f"{k_folds} folds created successfully.")

    fold_dirs = [os.path.join(output_dir, f"Fold_{fold+1}") for fold in range(k_folds)]
    return output_dir, fold_dirs


# Training step
def train_step_fold(model: torch.nn.Module,
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
        for batch, (X, y) in enumerate((data_loader)):
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
def test_step_fold(model: torch.nn.Module,
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

                # Append predictions and targets
                all_preds.extend(test_pred_labels.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

                # Update tqdm progress bar
                pbar.postfix = f"Loss: {test_loss/(batch+1):.4f} | Accuracy: {(test_acc/(batch+1))*100:.4f}%"
                pbar.update(1)

        # Adjust metrics to get the average loss and accuracy
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)


    return test_loss, test_acc, all_preds, all_targets


# Training and testing the model
def fold_train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          test_loader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device,
          k_folds,
          scheduler: torch.optim.lr_scheduler = None):
    """ Train the model and evaluate on the test set."""

    

    # Track the losses and accuracies
    results = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'train_all_preds': [[]],
        'train_all_targets': [[]],
        'test_all_preds': [[]],
        'test_all_targets': [[]]
        
    }
    # Train the model
    for epoch in range(epochs):
        # Train the model on the training set
        train_loss, train_acc, train_all_preds, train_all_targets = train_step_fold(model=model,
                                                                            data_loader=train_loader,
                                                                            loss_fn=loss_fn,
                                                                            optimizer=optimizer,
                                                                            device=device,
                                                                            ep=epoch,
                                                                            EPOCHS=epochs)


        # Evaluate the model on the test set
        test_loss, test_acc, test_all_preds, test_all_targets = test_step_fold(model=model,
                                                                            data_loader=test_loader,
                                                                            loss_fn=loss_fn,
                                                                            device=device,
                                                                            ep=epoch,
                                                                            EPOCHS=epochs)
        
        
        
        # Append the results
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)
        results['train_all_preds'].append(train_all_preds)
        results['train_all_targets'].append(train_all_targets)
        results['test_all_preds'].append(test_all_preds)
        results['test_all_targets'].append(test_all_targets)


    # Step the scheduler
    if scheduler:
        scheduler.step()

    return results
        
