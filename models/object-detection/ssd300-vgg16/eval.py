import torch
import cv2
import os
import numpy as np
from tqdm import tqdm
from config import DEVICE, NUM_CLASSES, NUM_WORKERS
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from model import create_model
from datasets import create_valid_dataset, create_valid_loader

#draw boxes
def draw_boxes(image, boxes, labels, scores, label_map, threshold=0.5):
    """
    Draws the bounding boxes on the image.

    Args:
        image (ndarray): The image to draw on.
        boxes (tensor): Bounding boxes coordinates.
        labels (tensor): Predicted labels.
        scores (tensor): Confidence scores for each box.
        label_map (dict): A dictionary mapping label indices to label names.
        threshold (float): Minimum confidence score to display the box.
    """
    image = np.array(image)  # Convert PIL image to NumPy array
    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            # Convert tensor to numpy
            box = box.cpu().numpy().astype(int)
            label = int(label.cpu().numpy())
            score = float(score.cpu().numpy())
            
            # Draw bounding box
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            text = f"{label_map[label]}: {score:.2f}"
            
            # Put label text above the box
            cv2.putText(image, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return image

# Evaluation function
def validate(valid_data_loader, model):
    print('Validating')
    model.eval()
    
    # Initialize tqdm progress bar.
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    target = []
    preds = []
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            outputs = model(images, targets)

        # For mAP calculation using Torchmetrics.
        #####################################
        for i in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()
            preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
            preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
            preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
            preds.append(preds_dict)
            target.append(true_dict)
            img_with_boxes = draw_boxes(images[i].cpu().permute(1, 2, 0),  # Convert from tensor format (C, H, W) to (H, W, C)
                                        preds_dict['boxes'], 
                                        preds_dict['labels'], 
                                        preds_dict['scores'], 
                                        ["bus", "truck", "motorcycle", "car"])
            
            # Save image to save directory
            save_path = os.path.join("./pred_imgs", f"image_{i}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
        #####################################

    metric = MeanAveragePrecision()
    metric.update(preds, target)
    metric_summary = metric.compute()
    precision = metric_summary['map_50']  # Precision at IoU=0.5
    recall = metric_summary['mar_100']    # Recall at IoU=0.5
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return metric_summary, precision, recall, f1_score

if __name__ == '__main__':
    # Load the best model and trained weights.
    model = create_model(num_classes=NUM_CLASSES, size=640)
    model.to(DEVICE).eval()

    test_dataset = create_valid_dataset(
        'C:/Users/User/CAM-R/models/object-detection/voc_data/test'
    )
    test_loader = create_valid_loader(test_dataset, num_workers=NUM_WORKERS)

    metric_summary, precision, recall, f1_score = validate(test_loader, model)
    print(f"mAP_50: {metric_summary['map_50']*100:.3f}")
    print(f"mAP_50_95: {metric_summary['map']*100:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1_score:.3f}")