from helper_functions import *
from classes import *
from torch.utils.data import DataLoader

TRAIN_DIR = 'data/Training'
TEST_DIR = 'data/Testing'
BATCH_SIZE = 32
IN_CHANNELS = 3
HIDDEN_UNITS = 32
NUM_CLASSES = 4
SIZE = 456
LEARNING_RATE = 0.0001
EPOCHS = 15
GAMMA = 0.1
STEP_SIZE = 7

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
model = MRI_classification_CNN(IN_CHANNELS, HIDDEN_UNITS, NUM_CLASSES, SIZE)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# Start the timer
start = timer()

# Train the model
results = train(model, train_loader, test_loader, criterion, optimizer, epochs=EPOCHS, device=DEVICE,
                scheduler=scheduler)

# End the timer
end = timer()
print_train_time(start, end, device=DEVICE)

model_dir = save_model(model, "cnn_model.pth", acc=results['train_loss'].__getitem__(-1), bs=BATCH_SIZE,
                       lr=LEARNING_RATE, ep=EPOCHS, sz=SIZE, hu=HIDDEN_UNITS)

# Plot the results
plot_loss_curves(results, model_dir=model_dir)

classes = train_dataset.classes

# Plot Distribution of data
plot_category_distribution(train_dataset, test_dataset, model_dir=model_dir)

# Plot accuracy per class
plot_accuracy_per_class(results, classes=classes, model_dir=model_dir)
