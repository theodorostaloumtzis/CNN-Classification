import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix



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
def plot_loss_curves(results, model_dir=None, fold=None):
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
        if fold:
            plt.savefig(os.path.join(model_dir, f"loss_fold_{fold}.png"))
        else:
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
        if fold:
            plt.savefig(os.path.join(model_dir, f"accuracy_fold_{fold}.png"))
        else:
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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def sensitivity_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sensitivity = np.diag(cm) / np.sum(cm, axis=1)
    specificity = []
    for i in range(cm.shape[0]):
        true_negatives = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        false_positives = np.sum(np.delete(cm, i, axis=0)[:, i])
        true_positives = cm[i, i]
        false_negatives = np.sum(np.delete(cm, i, axis=1)[i, :])
        specificity.append(true_negatives / (true_negatives + false_positives))
    return sensitivity, specificity

def plot_confusion_matrix_with_sensitivity(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    sensitivity, _ = sensitivity_specificity(np.argmax(cm, axis=1), np.argmax(cm, axis=0))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap=cmap, fmt='.2f', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



