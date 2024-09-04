import os
from PIL import Image
import numpy as np
import logging
import torch
import cv2
import pandas as pd
from super_gradients.training import models

# Remove all Logger notifications
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

# Load the YOLO model
yolo_nas_l = models.get("yolo_nas_l", pretrained_weights="coco")
include_labels = [2, 3, 5, 7]  # Example labels that are accepted

ROOT_DIR = "C:/Users/Jess/Desktop/School/FYP/one-day-test-images/testImages"
ROOT_DIR = "D:/CAM-R/images/images"

def grid_counting(camera_id):
    # Load the image to get its dimensions
    image_path = get_first_file(ROOT_DIR + "/" + camera_id)
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape

    # Define grid dimensions
    num_rows = 10
    num_cols = 10
    x_step = image_width / num_cols
    y_step = image_height / num_rows

    # Initialize the grid count matrix
    grid_counts = np.zeros((num_rows, num_cols))

    return grid_counts, num_rows, num_cols, x_step, y_step

def get_first_file(directory):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Get the first file
    if len(files) == 0:
        print("No files found in the directory.")
        return None
    
    first_file = files[0]
    
    # Construct full file path
    file_path = os.path.join(directory, first_file)
    
    return file_path

def apply_yolo_nas_l(image_path, grid_counts, num_rows, num_cols, x_step, y_step, post_200_images_processing, relevant_grids):
    image = Image.open(image_path)
    filtered_image = yolo_nas_l.predict(image, conf=0.25)
    pred = filtered_image.prediction
    labels = pred.labels.astype(int)

    xy_array = []
    updated_relevant_grids = set(relevant_grids)  # Create a copy to update

    for index, label in enumerate(labels):
        if label in include_labels:
            xmin, ymin, xmax, ymax = pred.bboxes_xyxy[index]

            # Calculate centroid
            cx = int((xmin + xmax) / 2)
            cy = int((ymin + ymax) / 2)

            # Determine grid cell based on the actual image dimensions
            col = min(max(int(cx // x_step), 0), num_cols - 1)
            row_idx = min(max(int(cy // y_step), 0), num_rows - 1)

            # Increment the count for the respective grid cell
            grid_counts[row_idx, col] += 1

            # Add the centroid and bounding box to the array
            xy_array.append([cx, cy, xmin, ymin, xmax, ymax])

    if post_200_images_processing:
        # Check grids and update relevant_grids
        for row_idx, col in list(updated_relevant_grids):
            if grid_counts[row_idx, col] >= minimum_collected:
                if grid_counts[row_idx, col] > maximum_collected:
                    grid_counts[row_idx, col] = maximum_collected  # Cap the count to maximum
                continue
            else:
                updated_relevant_grids.discard((row_idx, col))

    return xy_array, grid_counts, updated_relevant_grids

def calc_centroids(xy_array):
    centroids_arr = []
    centroids_and_box = []
    for box in xy_array:
        xmin, ymin, xmax, ymax = box[2], box[3], box[4], box[5]
        cx, cy = box[0], box[1]
        centroids_arr.append([cx, cy])
        centroids_and_box.append([[cx, cy], [xmin, ymin, xmax, ymax]])
    return centroids_arr, centroids_and_box

def save_csv(camera_id, file_path, centroids_and_box):
    flattened_data = [[cen_x, cen_y, xmin, ymin, xmax, ymax] for [cen_x, cen_y], [xmin, ymin, xmax, ymax] in centroids_and_box]
    df = pd.DataFrame(flattened_data, columns=['cen_x', 'cen_y', 'xmin', 'ymin', 'xmax', 'ymax'])
    df.to_csv(file_path + camera_id + ".csv", index=False)

# Main processing loop
camera_ids = ["BUKIT TIMAH EXPRESSWAY/2705"]

processed_images = 0
post_200_images_processing = False
relevant_grids = set()

# Accumulate centroids and bounding boxes
all_centroids_and_box = []

# Parameters
minimum_preprocess = 7
minimum_collected = 65
maximum_collected = 200

for camera_id in camera_ids:
    # Get grid setup
    grid_counts, num_rows, num_cols, x_step, y_step = grid_counting(camera_id)

    # Process each image
    for filename in os.listdir(ROOT_DIR + "/" + camera_id):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(ROOT_DIR + "/" + camera_id, filename)
            
            try:
                xy_array, grid_counts, updated_relevant_grids = apply_yolo_nas_l(
                    image_path, grid_counts, num_rows, num_cols, x_step, y_step, post_200_images_processing, relevant_grids
                )
                
                # Update relevant grids
                relevant_grids = updated_relevant_grids

                # Count processed images
                processed_images += 1
                # print(f"Processed images: {processed_images}")
            except Exception as e:
                print(e)
            
            if processed_images == 200:
                # After 200 images, identify relevant grids
                post_200_images_processing = True
                for row_idx in range(num_rows):
                    for col in range(num_cols):
                        if grid_counts[row_idx, col] >= minimum_preprocess:
                            relevant_grids.add((row_idx, col))
                print(xy_array)
                print(f"Relevant grids after 200 images: {relevant_grids}")

            if (post_200_images_processing and len(relevant_grids) == 0) or processed_images==1000:     # hard cap to see if its working
                print("All relevant grids have collected at least 65 centroids. Ending collection.")
                break

            # Accumulate centroids and bounding boxes
            centroids_arr, centroids_and_box = calc_centroids(xy_array)
            all_centroids_and_box.extend(centroids_and_box)

# Final save after processing all images
if all_centroids_and_box:
    save_csv(camera_ids[0].split("/")[1], "C:/Users/Jess/Desktop/School/FYP/CAM-R/models/experiments/lane_detection/centroidsuni/", all_centroids_and_box)

print(f"Total number of images processed: {processed_images}")
