import os
from PIL import Image
import numpy as np
import logging
import torch
import pandas as pd
import cv2
from super_gradients.training import models

# Remove all Logger notifications
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

yolo_nas_l = models.get("yolo_nas_l", pretrained_weights="coco")
include_labels = ["car", "bus", "truck", "motorcycle"]

ROOT_DIR = "C:/Users/Jess/Desktop/School/FYP/one-day-test-images/testImages"
ROOT_DIR = "D:/CAM-R/images/images"

os.chdir(ROOT_DIR)

pre_image_min = 300
pre_min_grid = 3
min_per_grid = 55
max_per_grid = 200 
total_image_process = 1300 # hard cap for the algorithm to stop running

def apply_yolo_nas_l(image_path, grid_counts, num_rows, num_cols, x_step, y_step, total_centroid_count):
    os.chdir(ROOT_DIR + "/" + image_path)

    accepted_list = [2, 3, 5, 7]  # Example labels that are accepted
    xy_array = []
    grid_indexes = []
    grid_check = False

    # Initialize image counter
    image_count = 0

    for filename in os.listdir("."):
        if image_count == pre_image_min: 
            for row_idx in range(grid_counts.shape[0]):
                for col_idx in range(grid_counts.shape[1]):
                    if grid_counts[row_idx, col_idx] >= pre_min_grid:
                        grid_indexes.append((row_idx, col_idx))
            print(grid_indexes)
            grid_check = True
            
        #### CHECK #######
        if image_count % 100 == 0:
            print(grid_counts)
            print(image_count, grid_indexes)
        try:
            image_count += 1
            image_path = os.path.join(".", filename)
            image = Image.open(image_path)
            filtered_image = yolo_nas_l.predict(image, conf=0.25)
            pred = filtered_image.prediction
            labels = pred.labels.astype(int)

            for index, label in enumerate(labels):
                if label in accepted_list:
                    xmin, ymin, xmax, ymax = pred.bboxes_xyxy[index]

                    # Calculate centroid
                    cx = int((xmin + xmax) / 2)
                    cy = int((ymin + ymax) / 2)

                    # Determine grid cell
                    col = min(max(int(cx // x_step), 0), num_cols - 1)
                    row_idx = min(max(int(cy // y_step), 0), num_rows - 1)
                    
                    if len(grid_indexes) == 0 and grid_check is False:
                        grid_counts[row_idx, col] += 1
                        
                        total_centroid_count += 1

                        # Add the centroid and bounding box to the array
                        xy_array.append([cx, cy, xmin, ymin, xmax, ymax])
                    
                    elif grid_check is True:
                        if ((row_idx, col) not in grid_indexes) or grid_counts[row_idx, col] >= max_per_grid:
                            continue
                        grid_counts[row_idx, col] += 1
                        total_centroid_count += 1
                        xy_array.append([cx, cy, xmin, ymin, xmax, ymax])
                        
                        if (grid_counts[row_idx, col] >= min_per_grid):
                            grid_indexes.remove((row_idx, col))
                            
                        if (grid_check is True and len(grid_indexes) == 0) or (image_count >= total_image_process):
                            # break
                            print(grid_counts)
                            xy_array, total_centroid_count, image_count
        except Exception as e:
            print(e)
            print(grid_indexes)
            
        # if (grid_check is True and len(grid_indexes) == 0) or (image_count >= total_image_process):
        #         break

    print(grid_counts)
    return xy_array, total_centroid_count, image_count

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

# List of camera IDs
camera_ids = [
    "BUKIT TIMAH EXPRESSWAY/2708"
]

# Initialize a total centroid count and image count
total_centroid_count = 0
total_image_count = 0

for camera_id in camera_ids:
    # Get grid setup
    grid_counts, num_rows, num_cols, x_step, y_step = grid_counting(camera_id)

    # Apply YOLO-NAS and get centroids
    xy_array, total_centroid_count, image_count = apply_yolo_nas_l(camera_id, grid_counts, num_rows, num_cols, x_step, y_step, total_centroid_count)

    # Update total image count
    total_image_count += image_count

    # Calculate centroids and save CSV
    centroids_arr, centroids_and_box = calc_centroids(xy_array)
    save_csv(camera_id.split("/")[1], "C:/Users/Jess/Desktop/School/FYP/CAM-R/models/experiments/lane_detection/centroidsuni/", centroids_and_box)
    
    print(f"Done w/ {camera_id}. Total Centroids Collected: {total_centroid_count}. Images Processed: {image_count}")
