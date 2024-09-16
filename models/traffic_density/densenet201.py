import torch
import torch.nn as nn
from torchvision.models import densenet201, DenseNet201_Weights
import td_load_data
import td_run_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DenseNet201(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet201, self).__init__()
        # Load a pre-trained DenseNet201 model and replace the classifier
        self.base_model = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
        num_ftrs = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.base_model(x)

if __name__ == "__main__":
    # Hyperparameters
    num_classes = 3  # Adjust this to match the number of classes (e.g., high, medium, low)
    num_epochs = 12
    batch_size = 32
    learning_rate = 0.001

    # Model initialization
    model = DenseNet201(num_classes=num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load data
    train_loader, validation_loader, test_loader, classes = td_load_data.create_data(batch_size=batch_size)
    
    # Train and validate the model
    td_run_model.train(num_epochs, device, model, criterion, optimizer, train_loader, validation_loader)
    
    # Save the model
    torch.save(model.state_dict(), 'densenet201_traffic_classifier.pth')
    
    # Test the model
    td_run_model.test(device, model, test_loader)
