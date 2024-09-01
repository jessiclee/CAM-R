from ultralytics import YOLO

model = YOLO("yolov8n.pt") 
metrics = model.val(data="C:/Users/User/CAM-R/models/object-detection/yolo_data/yolo_data.yaml", batch=32, iou=0.5, device="cpu")
# print(metrics)