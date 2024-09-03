from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt") 

# Train the model
results = model.train(data="C:/Users/User/CAM-R/models/object-detection/yolo_data/yolo_data.yaml", epochs=5, batch=32)