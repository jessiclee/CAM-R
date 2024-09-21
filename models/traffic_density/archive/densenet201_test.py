import torch
from torchvision.models import densenet201
import td_load_data
import td_run_model  # Assuming the function for testing is here

# Set the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model architecture with the same number of classes used during training
num_classes = 3  # Adjust this to match your dataset (ie. high, low, medium)
model = densenet201(num_classes=num_classes).to(device)

# Load the state dict
state_dict = torch.load('best_densenet201_model.pth')

# Remove the 'base_model.' prefix from the keys if it exists
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('base_model.'):
        name = k[len('base_model.'):]  # remove 'base_model.' from the key
    else:
        name = k
    new_state_dict[name] = v

# Load the modified state dict into the model
model.load_state_dict(new_state_dict)

model.eval()  # Set the model to evaluation mode

# Load the test data using the function in td_load_data.py
_, _, test_loader, classes = td_load_data.create_data(batch_size=32)

# Evaluate the model on the test set
td_run_model.test_model_on_test_data(device, model, test_loader)
