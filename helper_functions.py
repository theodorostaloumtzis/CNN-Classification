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
                pbar.postfix = f"Loss: {test_loss/(batch+1):.4f} | Accuracy: {(test_acc/(batch+1))*100:.4f}%"
                pbar.update(1)

        # Adjust metrics to get the average loss and accuracy
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)


    return test_loss, test_acc, all_preds, all_targets


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
        'all_preds': [],
        'all_targets': [],
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





        

