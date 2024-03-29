from helper_functions import *
from classes import *
from torch.utils.data import DataLoader
from torchvision import transforms

TRAIN_DIR = 'data/Training'
TEST_DIR = 'data/Testing'
BATCH_SIZE = 32
IN_CHANNELS = 3
HIDDEN_UNITS = 16  # Number of hidden units in the fully connected layer
NUM_CLASSES = 4
SIZE = 192
LEARNING_RATE = 0.001
EPOCHS = 2
GAMMA = 0.1
STEP_SIZE = 5
WEIGHT_DECAY = None
SEED = 42
EVAL_EPOCHS = 2

# Create the dictionary that hold the hyperparameters
hyperparameters = {
    "BATCH_SIZE": BATCH_SIZE,
    "IN_CHANNELS": IN_CHANNELS,
    "HIDDEN_UNITS": HIDDEN_UNITS,
    "NUM_CLASSES": NUM_CLASSES,
    "SIZE": SIZE,
    "LEARNING_RATE": LEARNING_RATE,
    "EPOCHS": EPOCHS,
    "EVAL_EPOCHS": EVAL_EPOCHS,  # "EVAL_EPOCHS": "Number of epochs to evaluate the model on the test set
    "GAMMA": GAMMA,
    "STEP_SIZE": STEP_SIZE,
    "WEIGHT_DECAY": WEIGHT_DECAY,
    "SEED": SEED
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the transforms
transform = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])

# Create the datasets
train_dataset = CustomDataset(TRAIN_DIR, transform=transform)
test_dataset = CustomDataset(TEST_DIR, transform=transform)

# Create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

from timeit import default_timer as timer

# Create the model
model = MRI_classification_CNN(IN_CHANNELS, HIDDEN_UNITS, NUM_CLASSES, SIZE).to(DEVICE)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# Set seed for reproducibility
set_seeds(SEED)

# Start the timer
start = timer()

# Train the model
results = train(model, train_loader, test_loader, criterion, optimizer, epochs=EPOCHS, device=DEVICE)

# End the timer
end = timer()
total_time = print_train_time(start, end, device=DEVICE)

# Evaluate the model
eval_results = evaluate(model, test_loader, criterion, device=DEVICE, eval_epochs=EVAL_EPOCHS)

model_dir = save_model(model, "MRI_classification_CNN", acc=eval_results['test_acc'],
                       hyperparameters=hyperparameters, total_time=total_time)

# Plot the results
plot_loss_curves(results, model_dir=model_dir)

classes = train_dataset.classes

# Plot accuracy per class
plot_accuracy_per_class(results, classes=classes, model_dir=model_dir)

# Plot confusion matrix
plot_confusion_matrix(eval_results, classes=classes, model_dir=model_dir)


# Plot the overall accuracy
plot_overall_accuracy_per_class(eval_results, classes=classes, model_dir=model_dir)
