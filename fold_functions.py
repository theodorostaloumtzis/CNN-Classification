import numpy as np
import os
import shutil
import torch
from tqdm.auto import tqdm
from PIL import Image
import pathlib


def combine_and_rename_images(data_dir, classes):
    # Create a combined directory for both training and testing sets

    combined_dir = 'Combined'
    os.makedirs(combined_dir, exist_ok=True)

    count_classes = {
        'glioma_tumor': 0,
        'meningioma_tumor': 0,
        'no_tumor': 0,
        'pituitary_tumor': 0
    }
    # Iterate through training and testing directories
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

    print("Images combined and renamed successfully.")
    return combined_dir

def split_to_train_test(data_dir, train_size, seed=None):
    # Create the training and testing directories
    dataset_dir = 'Dataset'
    train_dir = os.path.join(dataset_dir, 'Train')
    test_dir = os.path.join(dataset_dir, 'Test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    loaded_images_paths = {
        '0': [],
        '1': [],
        '2': [],
        '3': []
    }

    # Load the images
    for image in os.listdir(data_dir):
        class_idx = image.split('_')[0]
        image_path = os.path.join(data_dir, image)
        loaded_images_paths[class_idx].append(image_path)

    # Calculate the training and testing sizes
    train_len = {class_idx: int(train_size * len(loaded_images_paths[class_idx])) for class_idx in loaded_images_paths}

    
    # Move the images to the training directory
    for class_idx, image_paths in loaded_images_paths.items():
        count = 0 
        while train_len[class_idx] >= count:
            if seed:
                np.random.seed(seed)
            # Get a random image
            random_image = np.random.choice(image_paths)
            # Get the new image path
            new_image_path = os.path.join(train_dir, os.path.basename(random_image))
            # Move the image to the training directory
            shutil.move(random_image, new_image_path)
            # Remove the image from the list
            image_paths.remove(random_image)
            count += 1

    # Move the remaining images to the testing directory
    remaining_images = os.listdir(data_dir)
    for image in remaining_images:
        image_path = os.path.join(data_dir, image)
        new_image_path = os.path.join(test_dir, image)
        shutil.move(image_path, new_image_path)

    print("Images split into training and testing sets successfully.")
    return dataset_dir, train_dir, test_dir


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
            pbar.postfix = f"Loss: {train_loss/(batch+1):.4f} | Accuracy: {(train_acc/(batch+1))*100:.2f}%"
            pbar.update(1)

    # Calculate average loss and accuracy
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    return train_loss, train_acc, all_preds, all_targets


# Testing step
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

                # Append predictions and targets
                all_preds.extend(test_pred_labels.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

                # Update tqdm progress bar
                pbar.postfix = f"Loss: {test_loss/(batch+1):.4f} | Accuracy: {(test_acc/(batch+1))*100:.2f}%"
                pbar.update(1)

        # Adjust metrics to get the average loss and accuracy
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)


    return test_loss, test_acc, all_preds, all_targets


# Training and testing the model
def train_fold(model: torch.nn.Module,
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
        'train_all_preds': [[]],
        'train_all_targets': [[]],
        'test_all_preds': [[] ],
        'test_all_targets': [[] ]
    }

    # Train the model
    for epoch in range(epochs):
        train_loss, train_acc, train_all_preds, train_all_targets = train_step(model=model,
                                                                                data_loader=train_loader,
                                                                                loss_fn=loss_fn,
                                                                                optimizer=optimizer,
                                                                                device=device,
                                                                                ep=epoch,
                                                                                EPOCHS=epochs)

        # Evaluate the model on the test set
        test_loss, test_acc, test_all_preds, test_all_targets = test_step(model=model,
                                                                            data_loader=test_loader,
                                                                            loss_fn=loss_fn,
                                                                            device=device,
                                                                            ep=epoch,
                                                                            EPOCHS=epochs)

        # Save the results
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

# Evaluate the model
def evaluate_fold(model: torch.nn.Module,
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
        'all_targets': []
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

    # Print the results
    print("Evaluation results")
    print(f"Test Loss: {eval_results['test_loss']:.4f} | Test Acc: {eval_results['test_acc']*100:.2f}%")

    return eval_results

def extract_elements(data_list, indices):
    """
    Extract elements from a list based on the given indices.

    Parameters:
        data_list (list): The list of data.
        indices (numpy.ndarray): The indices to extract.

    Returns:
        numpy.ndarray: An array containing the elements corresponding to the indices.
    """
    data_array = np.array(data_list)
    result_array = data_array[indices]
    return result_array


def load_data(dir, transform=None):

    # Load the image paths to numpy array
    image_paths = list(pathlib.Path(dir).glob('*.jpg'))

    # Create the data and label arrays
    labels = []
    data = []

    for i in range(len(image_paths)):
        image_path = image_paths[i]
        image = Image.open(image_path)
        class_idx = image_path.name.split('_')[0]
        labels.append(class_idx)
        if transform:
            image = transform(image)
        data.append(image)

    labels = np.array(labels).astype(int)

    return data, labels