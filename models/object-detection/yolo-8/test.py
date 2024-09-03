from ultralytics import YOLO
#note pytorch 2.3 is needed for this to work
yolo_weights = ["./yolov8n.pt", "./yolov8x.pt", "./yolov8l.pt"]

for w in yolo_weights:
    model = YOLO(w) 
    metrics = model.val(data="C:/Users/User/CAM-R/models/object-detection/yolo_data/yolo_data.yaml", batch=32, iou=0.5, device="cpu", save_json=True, plots=True)
# print(metrics)