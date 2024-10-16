import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from torchvision import transforms
from PIL import Image

class_dict = {
    0: 'High',
    1: 'Low',
    2: 'Medium',
}

class VGG16MultiClassClassifier(nn.Module):
    def __init__(self, num_classes):
        super(VGG16MultiClassClassifier, self).__init__()
        self.base_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        
        in_features = self.base_model.classifier[6].in_features
        
        self.base_model.classifier[6] = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        return x

    def predictDensity(self, img, device):
        model = self.base_model
        state_dict = torch.load('43.pth', map_location=torch.device('cpu'))
        
        # Adjust the state_dict keys if needed
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('base_model.', '')  # Remove 'base_model.' prefix
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)

        model.eval()  # Set the model to evaluation mode

        # Define the transformation for all datasets (no augmentation)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(device)
        
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)  # Get the predicted class

        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        return probs, class_dict[predicted.item()]
