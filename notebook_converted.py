import os
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary
from sklearn.metrics import roc_curve, roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix
from timeit import default_timer as timer
from datetime import timedelta
import sklearn.model_selection
import torch_directml  # Import torch_directml for DirectML support

# Importing custom helper functions and classes
from helper_functions import *
from classes import *
from fold_functions import *
from plot_functions import *

# Define the hyperparameters
DATA_DIR = 'data'  # Path to the data directory
BATCH_SIZE = 32  # Batch size for the dataloaders
IN_CHANNELS = 3  # Number of input channels
HIDDEN_UNITS = 16  # Number of hidden units in the fully connected layer
NUM_CLASSES = 4  # Number of classes in the dataset
SIZE = 224  # Size of the images
LEARNING_RATE = 0.001  # Learning rate for the optimizer
EPOCHS = 250  # Number of epochs to train the model
GAMMA = 0.1  # Multiplicative factor of learning rate decay
STEP_SIZE = 15  # Step size for the learning rate scheduler
WEIGHT_DECAY = None  # Weight decay for the optimizer
SEED = 1678737  # Seed for reproducibility
RANDOM_ROTATION = 10  # Random rotation for the images

# Initialize DirectML device
dml = torch_directml.device()

# Use DirectML device for training
DEVICE = dml

print(DEVICE)

# Create the dictionary that holds the hyperparameters
hyperparameters = {
    'BATCH_SIZE': BATCH_SIZE,
    'IN_CHANNELS': IN_CHANNELS,
    'HIDDEN_UNITS': HIDDEN_UNITS,
    'NUM_CLASSES': NUM_CLASSES,
    'SIZE': SIZE,
    'LEARNING_RATE': LEARNING_RATE,
    'EPOCHS': EPOCHS,
    'GAMMA': GAMMA,
    'STEP_SIZE': STEP_SIZE,
    'WEIGHT_DECAY': WEIGHT_DECAY,
    'SEED': SEED,
    'RANDOM_ROTATION': RANDOM_ROTATION,
    'DEVICE': DEVICE
}

# Define the transforms
transform = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.RandomRotation(RANDOM_ROTATION),
    transforms.ToTensor(),
])

# Define the classes
classes = {
    'no_tumor': 0,
    'meningioma_tumor': 1,
    'pituitary_tumor': 2,
    'glioma_tumor': 3
}

# Placeholder for model type
model_type = "MRI_classification_CNN"  # Update with actual model type if different

# Create a directory based on the model type
output_dir = f"./{model_type}"
os.makedirs(output_dir, exist_ok=True)

# Function to save plots
def save_plot(plot_data, filename, output_dir):
    plt.figure()
    plt.plot(plot_data)
    plt.title(filename)
    plt.savefig(os.path.join(output_dir, f"{filename}.png"))
    plt.close()

# Function to save the model
def save_model(model, output_dir):
    model_save_path = os.path.join(output_dir, "trained_model.pth")
    torch.save(model.state_dict(), model_save_path)

# Function to save the model summary
def save_model_summary(model, input_size, output_dir):
    model_summary = summary(model, input_size)
    with open(os.path.join(output_dir, "model_summary.txt"), "w") as f:
        f.write(str(model_summary))

# Walk through directory and pre-process the dataset
walk_through_dir(DATA_DIR)
combined_dir = combine_and_rename_images(DATA_DIR, classes)
walk_through_dir(combined_dir)

# Load and split the dataset
dataset_data, dataset_labels = load_data('Combined', transform=transform)
shutil.rmtree('Combined')
train_data, test_data, train_labels, test_labels = sklearn.model_selection.train_test_split(dataset_data, dataset_labels, test_size=0.1, shuffle=True)
train_data, val_data, train_labels, val_labels = sklearn.model_selection.train_test_split(train_data, train_labels, test_size=0.12, shuffle=True)

# Print dataset sizes
print(len(train_data), len(train_labels))
print(len(val_data), len(val_labels))
print(len(test_data), len(test_labels))

# Create the dataloaders
train_loader = DataLoader(CustomDataset(train_data, train_labels), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(CustomDataset(val_data, val_labels), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(CustomDataset(test_data, test_labels), batch_size=BATCH_SIZE, shuffle=False)

# # Initialize the model and move it to DirectML device
# model = MRI_classification_CNN(IN_CHANNELS, HIDDEN_UNITS, NUM_CLASSES, SIZE, 0.2).to(DEVICE)


######################################################################################################################################################################

# Load the pre-trained EfficientNet model
model = models.efficientnet_b4(pretrained=True)  # Use efficientnet_b1, b2, etc., if needed

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Get the input features of the final Linear layer
in_features = model.classifier[1].in_features

# Modify the last fully connected layer
model.classifier = nn.Sequential(
    nn.Linear(in_features=in_features, out_features=NUM_CLASSES)
)

# Enable gradients for the parameters of the new fully connected layer
for param in model.classifier.parameters():
    param.requires_grad = True

# Unfreeze and reset the parameters of the last convolutional layer in `features`
# EfficientNet has a different structure; we'll assume the last conv layer is in the `features` module
for name, param in model.features[-1][-1].named_parameters():
    param.requires_grad = True
    if 'weight' in name:
        nn.init.kaiming_normal_(param)  # Initialize weights with Kaiming He initialization
    elif 'bias' in name:
        nn.init.zeros_(param)  # Initialize biases to zero

model = model.to(DEVICE)

######################################################################################################################################################################

# Define the optimizer and the loss function
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

# Define the scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# Training
start = timer()
results = train(model, train_loader, val_loader, loss_fn, optimizer, EPOCHS, DEVICE, scheduler=scheduler, early_stopping=False)
end = timer()

# Calculate and print training time
elapsed_time = end - start
formatted_time = str(timedelta(seconds=elapsed_time))
print(f'Training time: {formatted_time}')

# Evaluate the model
eval_res = evaluate(model, test_loader, loss_fn, DEVICE)

# Save the model and its summary
save_model(model, output_dir)
save_model_summary(model, (IN_CHANNELS, SIZE, SIZE), output_dir)

# Generate predictions and evaluate metrics
y_true = eval_res['all_targets']
y_pred_continuous = eval_res['all_preds']
y_pred = np.argmax(y_pred_continuous, axis=1)

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix_with_sensitivity(cm, classes, normalize=True)

# Plot loss curves
plot_loss_curves(results)

# Save plots
save_plot(results['train_loss'], 'Training_Loss', output_dir)
save_plot(results['val_loss'], 'Validation_Loss', output_dir)

# Generate additional metrics and plots
idx_to_class = {v: k for k, v in classes.items()}
all_preds = np.array(eval_res['all_preds'])
all_labels = np.array(eval_res['all_targets'])

print(f'all_preds shape: {all_preds.shape}')  # Should be (num_samples, NUM_CLASSES)
print(f'all_labels shape: {all_labels.shape}')  # Should be (num_samples,)

# Calculate ROC and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(all_labels == i, all_preds[:, i])
    roc_auc[i] = roc_auc_score(all_labels == i, all_preds[:, i])

# Calculate sensitivity, precision, and F1 score for each class
sensitivities = recall_score(all_labels, np.argmax(all_preds, axis=1), average=None)
precisions = precision_score(all_labels, np.argmax(all_preds, axis=1), average=None)
f1_scores = f1_score(all_labels, np.argmax(all_preds, axis=1), average=None)

# Print the metrics with class names
for i in range(len(classes)):
    class_name = idx_to_class[i]
    print(f'Class {class_name} - AUC: {roc_auc[i]:.2f}, Precision: {precisions[i]:.2f}, '
          f'Recall (Sensitivity): {sensitivities[i]:.2f}, F1 Score: {f1_scores[i]:.2f}')

# Plotting the ROC curve for each class
plt.figure()
for i in range(len(classes)):
    plt.plot(fpr[i], tpr[i], lw=2, label='Class {0} (AUC = {1:0.2f})'.format(idx_to_class[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Multiclass')
plt.legend(loc="lower right")
plt.savefig(os.path.join(output_dir, "ROC_Curve.png"))
plt.show()
