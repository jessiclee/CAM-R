# -*- coding: utf-8 -*-
"""YOLO NAS Centroids Generator

SUMMARY: 
Vehicle Detection using YOLO NAS with our trained weights.
We will be saving the coordinates along with their vehicle class.

IMPORTANT NOTICE: 
Only use python3.10, no more no less
super-gradients==3.7.1 Version

DEPENDENCIES: os, PIL, numpy, pandas, logging, torch, super_gradients

"""
##############################################################################
###########################  IMPORTS  ########################################
##############################################################################
import os
from PIL import Image
import numpy as np
import pandas as pd
import logging
import torch
from super_gradients.training import models

##############################################################################
########################### VARIABLES ########################################
##############################################################################

camera_ids = [
                "2703"
                ]

# Remove all Logger notifications
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

# Setting up YOLO NAS model
yolo_nas_l = models.get('yolo_nas_s', num_classes=4, checkpoint_path="C:/Users/Zhiyi/Desktop/FYP/CAM-R/models/experiments/lane_detection/contourHoughMethod/19.pth")
include_labels = ["car", "bus", "truck", "motorcycle"]

# Change directory to the folder containing the images
ROOT_DIR = "C:/Users/Zhiyi/Desktop/FYP/newtraffic/centroidimages"
os.chdir(ROOT_DIR)

##############################################################################
########################### FUNCTIONS ########################################
##############################################################################

def apply_yolo_nas_l(image_path):
    os.chdir(ROOT_DIR + "/" + image_path)

    # List of accepted labels
    accepted_list = [ 3,4,6,8]  # Example labels that are accepted

    xy_array = []
    # Apply the YOLO-NAS model to each image
    for filename in os.listdir("."):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust based on your image file types
            image_path = os.path.join(".", filename)
            image = Image.open(image_path)
            filtered_image = yolo_nas_l.predict(image, conf=0.25)
            # print(filtered_image)
            filtered_detections = []
            bboxes = []
            class_names = []
            class_indx = []
            conf = []
            pred = filtered_image.prediction
            labels = pred.labels.astype(int)

            for index, label in enumerate(labels):
                # print(label)
                if label in accepted_list:
                    # print(pred.bboxes_xyxy[index])
                    bboxes.append(pred.bboxes_xyxy[index])
                    class_indx.append(label)
                    conf.append(pred.confidence.astype(float)[index])

            # Update the filtered image with filtered detections
            pred.bboxes_xyxy = np.array(bboxes)
            xy_array.append(pred.bboxes_xyxy)
    return xy_array

def calc_centroids(xy_array):
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

def save_csv(camera_id, file_path, centroids_and_box):
    # Flatten the array and create a DataFrame
    flattened_data = [[cen_x, cen_y, xmin, ymin, xmax, ymax] for [cen_x, cen_y], [xmin, ymin, xmax, ymax] in centroids_and_box]

    # Create the DataFrame with headers
    df = pd.DataFrame(flattened_data, columns=['cen_x', 'cen_y', 'xmin', 'ymin', 'xmax', 'ymax'])

    # Save to CSV
    df.to_csv(file_path + camera_id + ".csv", index=False)


##############################################################################
###########################DRIVER CODE########################################
##############################################################################

for camera_id in camera_ids:
    xy_array = apply_yolo_nas_l(camera_id)
    centroids_arr, centroids_and_box = calc_centroids(xy_array)
    # save_csv(camera_id, "/content/drive/My Drive/FYP/csv/", centroids_and_box)
    save_csv(camera_id, "C:/Users/Zhiyi/Desktop/FYP/newtraffic/centroidimages/", centroids_and_box)
    print("Done w/ " + camera_id)