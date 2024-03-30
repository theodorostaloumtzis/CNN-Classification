from helper_functions import *
from classes import *
from torch.utils.data import DataLoader
from torchvision import transforms
from timeit import default_timer as timer

TRAIN_DIR = 'data/Training' # Path to the training directory
TEST_DIR = 'data/Testing' # Path to the testing directory
BATCH_SIZE = 32 # Batch size for the dataloaders
IN_CHANNELS = 3 # Number of input channels
HIDDEN_UNITS = 16  # Number of hidden units in the fully connected layer
NUM_CLASSES = 4 # Number of classes in the dataset
SIZE = 192 # Size of the images
LEARNING_RATE = 0.001 # Learning rate for the optimizer
EPOCHS = 15 # Number of epochs to train the model
GAMMA = 0.1 # Multiplicative factor of learning rate decay
STEP_SIZE = 5 # Step size for the learning rate scheduler
WEIGHT_DECAY = None # Weight decay for the optimizer
SEED = 42 # Seed for reproducibility
EVAL_EPOCHS = 10  # Number of epochs to evaluate the model on the test set
RANDOM_ROTATION = 10  # Random rotation for the images

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
    "SEED": SEED,
    "RANDOM_ROTATION": RANDOM_ROTATION
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the transforms
transform = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.RandomRotation(RANDOM_ROTATION),
    transforms.ToTensor()
])

# Create the datasets
train_dataset = CustomDataset(TRAIN_DIR, transform=transform)
test_dataset = CustomDataset(TEST_DIR, transform=transform)

# Create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

model_dir = save_model(model, acc=eval_results['test_acc'],
                       hyperparameters=hyperparameters, total_time=total_time)

# Plot the results
plot_loss_curves(results, model_dir=model_dir)

classes = train_dataset.classes

# Plot accuracy per class
plot_accuracy_per_class(results, classes=classes, model_dir=model_dir)

# Plot confusion matrix
plot_confusion_matrix(eval_results, classes=classes, model_dir=model_dir)
