import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def create_data(batch_size=32):
    # Define the directories for train, validation, and test datasets
    train_dir = "train"  # Path to training data
    # val_dir = "../split_data/val"      # Path to validation data
    test_dir = "C:/Users/Jess/OneDrive - Singapore Management University/FYP/midterm_demo/density"    # Path to test data

    # Define the transformation for all datasets (no augmentation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the datasets
    train_dataset = ImageFolder(root=train_dir, transform=transform)
    # val_dataset = ImageFolder(root=val_dir, transform=transform)
    test_dataset = ImageFolder(root=test_dir, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader, train_loader.classes
