import datetime
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import zipfile
from pathlib import Path
from tqdm.auto import tqdm
import requests
import seaborn as sns
from sklearn.metrics import confusion_matrix

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

    # Convert the total_time to a format hh:mm:ss
    total_time = str(datetime.timedelta(seconds=total_time))
    # Make seconds integer
    total_time = total_time.split(".")[0]
    print(f"Training time: {total_time}")
    return total_time


# Plot loss curves of a model
def plot_loss_curves(results, model_dir=None):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...],
             "train_all_preds": [...],
             "train_all_targets": [...],}
        model_dir (str): directory to save the plots, if None, plots will be displayed but not saved.
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.figure(figsize=(15, 7))
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid(True)
    if model_dir:
        plt.savefig(os.path.join(model_dir, "loss.png"))
        
    

    # Plot accuracy
    plt.figure(figsize=(15, 7))
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid(True)
    if model_dir:
        plt.savefig(os.path.join(model_dir, "accuracy.png"))


# Pred and plot image function from notebook 04 See creation:
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
def save_model(module, acc=None, hyperparameters=None, total_time=None):
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
            for key, value in hyperparameters.items():
                f.write(f"{key}: {value}\n")

    # Save the model
    model_path = os.path.join(model_dir, f"{model_name}.pt")
    torch.save(module.state_dict(), model_path)
    print(f"Model saved at {model_path}")
    return model_dir

# Plot the category distribution
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

# Find classes in a directory
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx

def finds_classes(targ_dirs: List[str]) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted(entry.name for targ_dir in targ_dirs for entry in os.scandir(targ_dir) if entry.is_dir())

    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {targ_dirs}.")

    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx

# Plot the accuracy per class
def plot_accuracy_per_class(results, classes=None, model_dir=None):
    """Plots the accuracy per class over epochs.

        Args:
            results (dict): Dictionary containing training and testing metrics.
            structure of results dictionary:
            results = {
                "train_loss": [list of training losses],
                "train_acc": [list of training accuracies],
                "test_loss": [list of testing losses],
                "test_acc": [list of testing accuracies],
                "train_all_preds": [list of all training predictions],
                "train_all_targets": [list of all training targets],
                "test_all_preds": [list of all testing predictions],
                "test_all_targets": [list of all testing targets],
            }

            classes (List[str], optional): List of class names.
            model_dir (str, optional): Directory to save the plot.
    """
    import numpy as np

    if results is None:
        print("Results dictionary is not available.")
        return
    elif classes is None:
        print("Classes are not available.")
        return
    elif model_dir is None:
        print("Model directory is not available.")
        return
    
    if results['train_all_preds'] is None or results['train_all_targets'] is None or results['test_all_preds'] is None or results['test_all_targets'] is None:
        print("Prediction and target data are not available.")
        return
    
    train_all_preds = results['train_all_preds']
    train_all_targets = results['train_all_targets']
    test_all_preds = results['test_all_preds']
    test_all_targets = results['test_all_targets']

    # Calculate accuracy per class per epoch for training and testing
    train_class_accuracy_per_epoch = [[0] * len(classes) for _ in range(len(train_all_preds))]
    test_class_accuracy_per_epoch = [[0] * len(classes) for _ in range(len(test_all_preds))]

    train_correct = [[0] * len(classes) for _ in range(len(train_all_preds))]
    test_correct = [[0] * len(classes) for _ in range(len(test_all_preds))]
    train_total =  [[0] * len(classes) for _ in range(len(train_all_preds))]
    test_total =  [[0] * len(classes) for _ in range(len(test_all_preds))]

    # Calculate accuracy per class per epoch for training 
    for epoch in range(len(train_all_preds)):
        for pred, target in zip(train_all_preds[epoch], train_all_targets[epoch]):
            train_correct[epoch][target] += (pred == target)
            train_total[epoch][target] += 1

        for i in range(len(classes)):
            # Check if train_total[epoch][i] is zero before division
            train_class_accuracy_per_epoch[epoch + 1][i] = train_correct[epoch + 1][i] / train_total[epoch + 1][i]
            

    # Calculate accuracy per class per epoch for testing
    for epoch in range(len(test_all_preds)):
        for pred, target in zip(test_all_preds[epoch], test_all_targets[epoch]):
            test_correct[epoch][target] += (pred == target)
            test_total[epoch][target] += 1

        for i in range(len(classes)):
            # Check if test_total[epoch][i] is zero before division
            test_class_accuracy_per_epoch[epoch + 1][i] = test_correct[epoch + 1][i] / test_total[epoch + 1][i]
            


    # Plot the accuracy per class for training and testing
    fig, axs = plt.subplots(2, figsize=(10, 10))
    fig.suptitle('Accuracy per Class Over Epochs')

    for i in range(len(classes)):
        axs[0].plot(range(len(train_class_accuracy_per_epoch)), [epoch[i] for epoch in train_class_accuracy_per_epoch], label=classes[i])
    axs[0].set_title('Training Accuracy per Class')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()
    axs[0].grid(axis='y')

    for i in range(len(classes)):
        axs[1].plot(range(len(test_class_accuracy_per_epoch)), [epoch[i] for epoch in test_class_accuracy_per_epoch], label=classes[i])
    axs[1].set_title('Testing Accuracy per Class')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(axis='y')

    plt.tight_layout()
    if model_dir:
        plt.savefig(model_dir + '/accuracy_per_class.png')
    plt.show()


def plot_confusion_matrix(results, classes=None, model_dir=None):
    """Plots and saves the confusion matrix.

    Args:
        results (dict): Dictionary containing training and testing metrics.
        classes (List[str], optional): List of class names.
        model_dir (str, optional): Directory to save the plot.
    """

    all_preds = results.get('all_preds', [])
    all_targets = results.get('all_targets', [])

    if not all_preds or not all_targets:
        print("Prediction and target data are not available.")
        return

    # Calculate confusion matrix
    cm = confusion_matrix(all_preds, all_targets)

    # Clacula accuracy per class
    class_accuracy = cm.diagonal() / cm.sum(axis=1)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)

    if classes:
        plt.xticks(ticks=np.arange(len(classes)) + 0.5, labels=classes)
        plt.yticks(ticks=np.arange(len(classes)) + 0.5, labels=classes)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')

    plt.title('Confusion Matrix')
    plt.tight_layout()

    # Save the plot if model directory is provided
    if model_dir:
        plt.savefig(os.path.join(model_dir, "confusion_matrix.png"))

    plt.show()

    # Plot the overall accuracy per class
    plt.figure(figsize=(10, 6))
    plt.bar(classes, class_accuracy)
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Class')
    plt.tight_layout()
    plt.ylim(0, 1)
    plt.grid(axis='y')

    if model_dir:
        plt.savefig(os.path.join(model_dir, 'accuracy_per_class.png'))
    
    plt.show()


        

