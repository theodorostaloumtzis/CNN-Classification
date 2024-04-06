from helper_functions import *
from classes import *
from torch.utils.data import DataLoader
from torchvision import transforms
from timeit import default_timer as timer
from fold_functions import *
from plot_functions import *
from sklearn.model_selection import StratifiedKFold

def select_model():
    """
    Select the model to train and evaluate.
    """
    print("Select the model to train and evaluate:")
    print("1. MRI_classification_CNN")
    print("2. EfficientNet")
    print("3. ResNet")
    model = int(input("Enter the model number: "))

    #do while to check input
    while model not in [1, 2, 3]:
        model = int(input("Invalid model number. Please select a valid model number: "))
    
    
    if model == 1:
        model_name = "MRI_classification_CNN"

    elif model == 2:
        model_name = input("Enter the model name(b0/b1/b2/b3/b4/b5/b6/b7): ")
        while model_name not in ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]:
            model_name = input("Invalid model name. Please select a valid model name: ")
    elif model == 3:
        
        model_name = input("Enter the model params(resnet18/resnet34/resnet50/resnet101/resnet152): ")
        while model_name not in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
            model_name = input("Invalid model name. Please select a valid model name: ")
    else:
        raise ValueError("Invalid model number. Please select a valid model number.")
    
    folding = input("Do you want to perform K-Fold Cross Validation? (y/n): ")

    while folding not in ["y", "n"]:
        folding = input("Invalid input. Please enter 'y' or 'n': ")

    return model, model_name, folding   

# Define the main function
def main():
    """
    Main function to train and evaluate the model.
    """
    model_type, model_name, folding = select_model()

    # Define the hyperparameters
    DATA_DIR = 'data' # Path to the data directory
    BATCH_SIZE = 32# Batch size for the dataloaders
    IN_CHANNELS = 3 # Number of input channels
    HIDDEN_UNITS = 16  # Number of hidden units in the fully connected layer
    NUM_CLASSES = 4 # Number of classes in the dataset
    SIZE = 224 # Size of the images
    LEARNING_RATE = 0.001 # Learning rate for the optimizer
    EPOCHS = 10 # Number of epochs to train the model
    K_FOLDS = 10 # Number of folds for K-Fold Cross Validation
    GAMMA = 0.1 # Multiplicative factor of learning rate decay
    STEP_SIZE = 6 # Step size for the learning rate scheduler
    WEIGHT_DECAY = None # Weight decay for the optimizer
    SEED = 1737 # Seed for reproducibility
    RANDOM_ROTATION = 10  # Random rotation for the images
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create the dictionary that hold the hyperparameters
    hyperparameters = {
        'BATCH_SIZE': BATCH_SIZE,
        'IN_CHANNELS': IN_CHANNELS,
        'HIDDEN_UNITS': HIDDEN_UNITS,
        'NUM_CLASSES': NUM_CLASSES,
        'SIZE': SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'EPOCHS': EPOCHS,
        'K_FOLDS': K_FOLDS,
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
        transforms.ToTensor()
    ])
    
    # Define the classes
    classes = {
    'no_tumor': 0,
    'meningioma_tumor': 1,
    'pituitary_tumor': 2,
    'glioma_tumor': 3
    }

    if folding  == 'n':
        # Pre-Process the dataset
        combine_dir = combine_and_rename_images(DATA_DIR, classes=classes)

        walk_through_dir(combine_dir)

        # Create the train and test directories
        dataset_dir, train_dir, test_dir = split_to_train_test(combine_dir, 0.9)

        # Delete the empty directory
        shutil.rmtree(combine_dir)

        # Load the dataset
        train_data, train_labels = load_data(train_dir, transform=transform)
        test_data, test_labels = load_data(test_dir, transform=transform)

        # Create the train and test loaders
        train_loader = DataLoader(CustomDataset(train_data, train_labels), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(CustomDataset(test_data, test_labels), batch_size=BATCH_SIZE, shuffle=False)

        epochs = int(input("Enter the number of epochs: "))
        while epochs < 1:
            epochs = int(input("Invalid number of epochs. Please enter a valid number of epochs: "))

        # Update the epochs hyperparameter
        hyperparameters['EPOCHS'] = epochs
        EPOCHS = epochs

        model = 0
        # Train the model
        if model_type == 1:
            model = MRI_classification_CNN(IN_CHANNELS, NUM_CLASSES, HIDDEN_UNITS, SIZE).to(DEVICE)
        elif model_type == 2:
            model = EfficientNet(model_name=model_name, output=NUM_CLASSES).to(DEVICE)

        elif model_type == 3:
            model = ResNet(resnet_name=model_name, in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(DEVICE)

        # Define the optimizer and the loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Define the loss function
        loss_fn = nn.CrossEntropyLoss()

        start = timer()

        results = train(model, 
                        train_loader, 
                        test_loader, 
                        loss_fn, 
                        optimizer, 
                        EPOCHS, 
                        DEVICE)

        end = timer()
        total_time = print_train_time(start, end)

        # Evaluate the model
        eval_res = evaluate(model, 
                            test_loader, loss_fn, DEVICE)

        # Save the model 
        model_dir = save_model(model, eval_res['test_acc'], hyperparameters=hyperparameters, total_time=total_time)

        # Plot the results
        plot_loss_curves(results, model_dir)

        # Plot the confusion matrix
        plot_confusion_matrix(eval_res, classes=classes, model_dir=model_dir)

        shutil.rmtree(dataset_dir)
        
    elif folding == 'y':

        # Pre-Process the dataset
        combine_dir = combine_and_rename_images(DATA_DIR, classes=classes)

        walk_through_dir(combine_dir)


        # Load the dataset
        dataset_data, dataset_labels = load_data(combine_dir, transform=transform)
        

        # Define the stratified K-Fold Cross Validation
        skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)


        fold_results = []
        fold_models = []
        fold_eval_results = []
        time_per_fold = []

        
        print("Training the model...")
        for fold, (train_idx, val_idx) in enumerate(skf.split(dataset_data, dataset_labels)):
            print(f"Fold combination: [{fold + 1}/{K_FOLDS}]")
            

            # Create the train and validation loaders
            train_fold_data = extract_elements(dataset_data, train_idx)
            train_fold_labels = extract_elements(dataset_labels, train_idx)
            val_fold_data = extract_elements(dataset_data, val_idx)
            val_fold_labels = extract_elements(dataset_labels, val_idx)

            train_loader = DataLoader(CustomDataset(train_fold_data, train_fold_labels), batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(CustomDataset(val_fold_data, val_fold_labels), batch_size=BATCH_SIZE, shuffle=False)
            
            model = 0
            # Set Seed
            set_seeds(SEED)
            # Define the model
            if model_type == 1:
                model = MRI_classification_CNN(IN_CHANNELS, NUM_CLASSES, HIDDEN_UNITS, SIZE).to(DEVICE)
            elif model_type == 2:
                model = EfficientNet(model_name=model_name, output=NUM_CLASSES).to(DEVICE)
            elif model_type == 3:
                model = ResNet(resnet_name=model_name, in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(DEVICE)

            # Define the optimizer and the loss function
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            # Define the loss function
            loss_fn = nn.CrossEntropyLoss()
            # Define the learning rate scheduler
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

            start = timer()
            res = train_fold(model, 
                            train_loader,
                            val_loader,
                            loss_fn,
                            optimizer,
                            EPOCHS,
                            DEVICE,
                            scheduler)
            
            end = timer()

            total_time = print_train_time(start, end)

            time_per_fold.append(total_time)

            fold_results.append(res)
            fold_models.append(model)

            # Evaluate the model
            eval_res = evaluate(model,
                                val_loader,
                                loss_fn,
                                DEVICE,
                                )
            
            fold_eval_results.append(eval_res)

        

        total_time = print_train_time(start, end)

        # Find the best model based on the test accuracy
        best_model_idx = np.argmax([res['test_acc'] for res in fold_eval_results])

        # Save the best model
        model_dir = save_model(fold_models[best_model_idx],
                            fold_eval_results[best_model_idx]['test_acc'],
                            hyperparameters=hyperparameters,
                            total_time=time_per_fold[best_model_idx],
                            fold=best_model_idx+1)
        
        # Plot the loss curves and the confusion matrix for the best model
        plot_loss_curves(fold_results[best_model_idx], model_dir, fold=best_model_idx+1)
        plot_confusion_matrix(fold_eval_results[best_model_idx], classes=classes, model_dir=model_dir, fold=best_model_idx+1)
        

        shutil.rmtree(combine_dir)
        

if __name__ == '__main__':
    main()
