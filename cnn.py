from helper_functions import *
from classes import *
from torch.utils.data import DataLoader
from torchvision import transforms
from timeit import default_timer as timer

def select_model():
    """
    Select the model to train and evaluate.
    """
    print("Select the model to train and evaluate:")
    print("1. MRI_classification_CNN")
    print("2. EfficientNet")
    model = int(input("Enter the model number: "))

    #do while to check input
    while model not in [1, 2]:
        model = int(input("Invalid model number. Please select a valid model number: "))
    
    
    if model == 1:
        model_name = "MRI_classification_CNN"

    elif model == 2:
        model_name = input("Enter the model name(b0/b1/b2/b3/b4/b5/b6/b7): ")
        while model_name not in ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]:
            model_name = input("Invalid model name. Please select a valid model name: ")
    else:
        raise ValueError("Invalid model number. Please select a valid model number.")

    return model, model_name

# Define the main function
def main():
    """
    Main function to train and evaluate the model.
    """
    model_type, model_name = select_model()

    # Define the hyperparameters
    DATA_DIR = 'data' # Path to the data directory
    BATCH_SIZE = 8 # Batch size for the dataloaders
    IN_CHANNELS = 3 # Number of input channels
    HIDDEN_UNITS = 16  # Number of hidden units in the fully connected layer
    NUM_CLASSES = 4 # Number of classes in the dataset
    SIZE = 224 # Size of the images
    LEARNING_RATE = 0.001 # Learning rate for the optimizer
    EPOCHS = 10 # Number of epochs to train the model
    K_FOLDS = 6 # Number of folds for K-Fold Cross Validation
    GAMMA = 0.1 # Multiplicative factor of learning rate decay
    STEP_SIZE = 6 # Step size for the learning rate scheduler
    WEIGHT_DECAY = 0.025 # Weight decay for the optimizer
    SEED = 42 # Seed for reproducibility
    EVAL_EPOCHS = 10  # Number of epochs to evaluate the model on the test set
    RANDOM_ROTATION = 10  # Random rotation for the images
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        "RANDOM_ROTATION": RANDOM_ROTATION,
        "DEVICE": DEVICE
    }

    # Define the transforms
    transform = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.RandomRotation(RANDOM_ROTATION),
        transforms.ToTensor()
    ])

    # Load the dataset
    train_dataset = CustomDataset('data/Training', transform=transform)
    test_dataset = CustomDataset('data/Testing', transform=transform)

    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create the model
    if model_type == 1:
        model = MRI_classification_CNN(IN_CHANNELS, HIDDEN_UNITS, NUM_CLASSES, SIZE).to(DEVICE)

    elif model_type == 2:
        model = EfficientNet(model_name, NUM_CLASSES).to(DEVICE)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # Train the model
    start = timer()

    results = train(model, train_loader, test_loader, criterion, optimizer, EPOCHS, DEVICE, scheduler)

    end = timer()

    # Print the results
    total_time = print_train_time(start, end)

    # Evaluate the model
    eval_res = evaluate(model, test_loader, criterion, DEVICE)

    # Save the model
    save_model(model, model_name, total_time, hyperparameters, results)





    
        

if __name__ == '__main__':
    main()
