import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader
from PIL import Image
import os

device: str = "cuda" if torch.cuda.is_available() else "cpu"

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_dir, transforms=None, size=(640, 640)):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(self.image_dir)))
        self.size = size

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.imgs[idx])
        ann_path = os.path.join(self.annotation_dir, self.imgs[idx].replace('.jpg', '.txt'))
        
        img = Image.open(img_path).convert("RGB")
        img = img.resize(self.size)

        boxes = []
        labels = []
        with open(ann_path) as f:
            for line in f.readlines():
                parts = line.strip().split(',')
                xmin, ymin, xmax, ymax, label = map(int, parts)
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

# Define transformations
transform = T.Compose([
    T.ToTensor(),
])

# Create the dataset with resizing
train_dataset = CustomDataset(
    image_dir='C:/Users/jesle/Desktop/fyp/actual_data/model_evaluation_data/train/images',
    annotation_dir='C:/Users/jesle/Desktop/fyp/actual_data/text_labels/train',
    transforms=transform,
    size=(640, 640)  # Fixed size
)

val_dataset = CustomDataset(
    image_dir='C:/Users/jesle/Desktop/fyp/actual_data/model_evaluation_data/valid/images',
    annotation_dir='C:/Users/jesle/Desktop/fyp/actual_data/text_labels/valid',
    transforms=transform,
    size=(640, 640)  # Fixed size
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

# Modify the model for the number of classes in your dataset
num_classes = 4  # Including background (e.g., 7 classes + 1 background)
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

model.to(device)

# Training parameters
num_epochs = 4
optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Initialize best validation loss and best model path
best_val_loss = float('inf')
best_model_path = 'best_model.pth'

def evaluate(model, data_loader, device):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        total_loss = 0
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

        avg_loss = total_loss / len(data_loader)
    return avg_loss

def main():
    print("Training start")

    for epoch in range(num_epochs):
        model.train()
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        # Step the learning rate scheduler
        lr_scheduler.step()

        # Validation
        val_loss = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item()}, Validation Loss: {val_loss}")

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")

    print("Training completed.")

if __name__ == '__main__':
    main()
