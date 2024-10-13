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

def predictDensity(num_classes, image_path, device):
    
    model = VGG16MultiClassClassifier(num_classes=num_classes).to(device)
    state_dict = torch.load('weights/43.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    model.eval()  # Set the model to evaluation mode

    # Define the transformation for all datasets (no augmentation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path)
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)
    
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs, 1)  # Get the predicted class

    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(outputs, dim=1)
    
    return probs, class_dict[predicted.item()]

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 3
# dynamically change in the web app after API got the image
image_path = "C:/Users/Jess/OneDrive - Singapore Management University/FYP/midterm_demo/density/medium/1503_15-07-2024_18-00-01.jpg" 
try:
    probs, predicted = predictDensity(num_classes, image_path, device)
    print(f"Predicted class: {predicted}, Probabilities: {probs}")
    # return predicted        # Classification found, return to web interface
except:
    print("Something went wrong.")
    # return None             # Something went wrong, no classification found
    