import numpy as np
import torch
from PIL import Image
from super_gradients.training import models
import pandas as pd 

class YOLO_NAS_L:
    def __init__(self, pth_file):
        self.model = models.get('yolo_nas_s', num_classes=4, checkpoint_path=pth_file)
    
    def predict(self, image):

        # List of accepted labels
        accepted_list = [0,1,2,3]  # Example labels that are accepted
        xy_array = []
        filtered_image = self.model.predict(image, conf=0.25, iou=0.1)

        bboxes = []
        class_indx = []
        conf = []
        pred = filtered_image.prediction
        labels = pred.labels.astype(int)
        for index, label in enumerate(labels):
            confi = pred.confidence.astype(float)[index]
            if (label==0 and confi > 0.6) or (label==1 and confi > 0.5) or (label==2 and confi > 0.65) or (label==3 and confi > 0.35):
                bboxes.append(pred.bboxes_xyxy[index])
                class_indx.append(label)
                conf.append(confi)
        bboxes, class_indx, conf = self.filter_high_iou_boxes(np.array(bboxes), np.array(class_indx), np.array(conf), 0.5)
        pred.bboxes_xyxy = np.array(bboxes) 
        pred.labels = np.array(class_indx)
        pred.confidence = np.array(conf)
        # Update the filtered image with filtered detections
        xy_array.append(np.array(bboxes))
        centroids_arr, centroids_and_box = self.calc_centroids(xy_array)
        df = self.get_centroids_df(centroids_and_box, class_indx)
            
        return df
    
    def filter_high_iou_boxes(self, boxes, classes, scores, iou_threshold=0.1):
        keep = []
        for i, box in enumerate(boxes):
            if all(self.calculate_iou(box, boxes[j]) < iou_threshold for j in keep):
                keep.append(i)
        return boxes[keep], classes[keep], scores[keep]
    
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Compute intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Compute area of each bounding box
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Compute union area
        union = area1 + area2 - intersection
        
        return intersection / union if union != 0 else 0
    
    def get_centroids_df(self, centroids_and_box, class_indx):
        # Flatten the array and create a DataFrame
        flattened_data = [[cen_x, cen_y, xmin, ymin, xmax, ymax] for [cen_x, cen_y], [xmin, ymin, xmax, ymax] in centroids_and_box]

        # Create the DataFrame with headers
        df = pd.DataFrame(flattened_data, columns=['cen_x', 'cen_y', 'xmin', 'ymin', 'xmax', 'ymax'])
        df["class"] = class_indx
        return df

    def calc_centroids(self, xy_array):
        centroids_arr = []
        centroids_and_box = []
        for image in xy_array:
            for box in image:
                # 0 - xmin, 1 - ymin, 2 - xmax, 3 - ymax
                xmin, ymin, xmax, ymax = box
                cx = int((xmin + xmax) / 2)
                cy = int((ymin + ymax) / 2)
                centroids_arr.append([cx, cy])
                centroids_and_box.append([[cx, cy] , [xmin, ymin, xmax, ymax]])
        return centroids_arr, centroids_and_box

