from ultralytics import YOLO
import torch

device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
print(f"device = {device}")

# Load a model
model = YOLO("yolov9c.pt")  # load a pretrained model (recommended for training)
model.to(device)

# Use the model
model.train(data="dataset.yaml", epochs=1, imgsz=640, save_period=1, plots=True)  # train the model
metrics = model.val()  # evaluate model performance on the validation set