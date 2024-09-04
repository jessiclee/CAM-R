# import os
# from PIL import Image
# import numpy as np
# import logging
# import torch
# import cv2
# import pandas as pd
# from super_gradients.training import models

# # Remove all Logger notifications
# logger = logging.getLogger()
# logger.setLevel(logging.CRITICAL)

# # Load the YOLO model
# yolo_nas_l = models.get("yolo_nas_l", pretrained_weights="coco")
# include_labels = [2, 3, 5, 7]  # Example labels that are accepted

# ROOT_DIR = "C:/Users/Jess/Desktop/School/FYP/one-day-test-images/testImages"
# ROOT_DIR = "D:/CAM-R/images/images"

# def grid_counting(camera_id):
#     # Load the image to get its dimensions
#     image_path = get_first_file(ROOT_DIR + "/" + camera_id)
#     image = cv2.imread(image_path)
#     image_height, image_width, _ = image.shape

#     # Define grid dimensions
#     num_rows = 10
#     num_cols = 10
#     x_step = image_width / num_cols
#     y_step = image_height / num_rows

#     # Initialize the grid count matrix
#     grid_counts = np.zeros((num_rows, num_cols))

#     return grid_counts, num_rows, num_cols, x_step, y_step

# def get_first_file(directory):
#     # List all files in the directory
#     files = os.listdir(directory)
    
#     # Get the first file
#     if len(files) == 0:
#         print("No files found in the directory.")
#         return None
    
#     first_file = files[0]
    
#     # Construct full file path
#     file_path = os.path.join(directory, first_file)
    
#     return file_path

# def apply_yolo_nas_l(image_path, grid_counts, num_rows, num_cols, x_step, y_step, post_200_images_processing, relevant_grids):
#     image = Image.open(image_path)
#     filtered_image = yolo_nas_l.predict(image, conf=0.25)
#     pred = filtered_image.prediction
#     labels = pred.labels.astype(int)

#     xy_array = []
#     updated_relevant_grids = set(relevant_grids)  # Create a copy to update

#     for index, label in enumerate(labels):
#         if label in include_labels:
#             xmin, ymin, xmax, ymax = pred.bboxes_xyxy[index]

#             # Calculate centroid
#             cx = int((xmin + xmax) / 2)
#             cy = int((ymin + ymax) / 2)

#             # Determine grid cell based on the actual image dimensions
#             col = min(max(int(cx // x_step), 0), num_cols - 1)
#             row_idx = min(max(int(cy // y_step), 0), num_rows - 1)

#             # Increment the count for the respective grid cell
#             grid_counts[row_idx, col] += 1

#             # Add the centroid and bounding box to the array
#             xy_array.append([cx, cy, xmin, ymin, xmax, ymax])

#     if post_200_images_processing:
#         # Check grids and update relevant_grids
#         for row_idx, col in list(updated_relevant_grids):
#             if grid_counts[row_idx, col] >= minimum_collected:
#                 if grid_counts[row_idx, col] > maximum_collected:
#                     grid_counts[row_idx, col] = maximum_collected  # Cap the count to maximum
#                 continue
#             else:
#                 updated_relevant_grids.discard((row_idx, col))

#     return xy_array, grid_counts, updated_relevant_grids

# def calc_centroids(xy_array):
#     centroids_arr = []
#     centroids_and_box = []
#     for box in xy_array:
#         xmin, ymin, xmax, ymax = box[2], box[3], box[4], box[5]
#         cx, cy = box[0], box[1]
#         centroids_arr.append([cx, cy])
#         centroids_and_box.append([[cx, cy], [xmin, ymin, xmax, ymax]])
#     return centroids_arr, centroids_and_box

# def save_csv(camera_id, file_path, centroids_and_box):
#     flattened_data = [[cen_x, cen_y, xmin, ymin, xmax, ymax] for [cen_x, cen_y], [xmin, ymin, xmax, ymax] in centroids_and_box]
#     df = pd.DataFrame(flattened_data, columns=['cen_x', 'cen_y', 'xmin', 'ymin', 'xmax', 'ymax'])
#     df.to_csv(file_path + camera_id + ".csv", index=False)

# # Main processing loop
# camera_ids = ["BUKIT TIMAH EXPRESSWAY/2705"]

# processed_images = 0
# post_200_images_processing = False
# relevant_grids = set()

# # Accumulate centroids and bounding boxes
# all_centroids_and_box = []

# # Parameters
# minimum_preprocess = 7
# minimum_collected = 65
# maximum_collected = 200

# for camera_id in camera_ids:
#     # Get grid setup
#     grid_counts, num_rows, num_cols, x_step, y_step = grid_counting(camera_id)

#     # Process each image
#     for filename in os.listdir(ROOT_DIR + "/" + camera_id):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             image_path = os.path.join(ROOT_DIR + "/" + camera_id, filename)
            
#             try:
#                 xy_array, grid_counts, updated_relevant_grids = apply_yolo_nas_l(
#                     image_path, grid_counts, num_rows, num_cols, x_step, y_step, post_200_images_processing, relevant_grids
#                 )
                
#                 # Update relevant grids
#                 relevant_grids = updated_relevant_grids

#                 # Count processed images
#                 processed_images += 1
#                 # print(f"Processed images: {processed_images}")
#             except Exception as e:
#                 print(e)
            
#             if processed_images == 200:
#                 # After 200 images, identify relevant grids
#                 post_200_images_processing = True
#                 for row_idx in range(num_rows):
#                     for col in range(num_cols):
#                         if grid_counts[row_idx, col] >= minimum_preprocess:
#                             relevant_grids.add((row_idx, col))
#                 print(xy_array)
#                 print(f"Relevant grids after 200 images: {relevant_grids}")

#             if (post_200_images_processing and len(relevant_grids) == 0) or processed_images==5:     # hard cap to see if its working
#                 print(xy_array)
#                 print("All relevant grids have collected at least 65 centroids. Ending collection.")
#                 break

#         # Accumulate centroids and bounding boxes
#         centroids_arr, centroids_and_box = calc_centroids(xy_array)
#         all_centroids_and_box.extend(centroids_and_box)

# # Final save after processing all images
# if all_centroids_and_box:
#     save_csv(camera_ids[0].split("/")[1], "C:/Users/Jess/Desktop/School/FYP/CAM-R/models/experiments/lane_detection/centroidsuni/", all_centroids_and_box)

# print(f"Total number of images processed: {processed_images}")

# import os
# from PIL import Image
# import numpy as np
# import logging
# import torch
# import pandas as pd
# import cv2
# from super_gradients.training import models

# # Remove all Logger notifications
# logger = logging.getLogger()
# logger.setLevel(logging.CRITICAL)

# yolo_nas_l = models.get("yolo_nas_l", pretrained_weights="coco")
# include_labels = ["car", "bus", "truck", "motorcycle"]

# ROOT_DIR = "C:/Users/Jess/Desktop/School/FYP/one-day-test-images/testImages"
# os.chdir(ROOT_DIR)

# def apply_yolo_nas_l(image_path, grid_counts, num_rows, num_cols, x_step, y_step, limit):
#     os.chdir(ROOT_DIR + "/" + image_path)

#     accepted_list = [2, 3, 5, 7]  # Example labels that are accepted
#     xy_array = []

#     for filename in os.listdir("."):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             image_path = os.path.join(".", filename)
#             image = Image.open(image_path)
#             filtered_image = yolo_nas_l.predict(image, conf=0.25)
#             pred = filtered_image.prediction
#             labels = pred.labels.astype(int)

#             for index, label in enumerate(labels):
#                 if label in accepted_list:
#                     xmin, ymin, xmax, ymax = pred.bboxes_xyxy[index]

#                     # Calculate centroid
#                     cx = int((xmin + xmax) / 2)
#                     cy = int((ymin + ymax) / 2)

#                     # Determine grid cell
#                     col = min(max(int(cx // x_step), 0), num_cols - 1)
#                     row_idx = min(max(int(cy // y_step), 0), num_rows - 1)

#                     if limit == -1:
#                         xy_array.append([cx, cy, xmin, ymin, xmax, ymax])
                    
#                     elif grid_counts[row_idx, col] < limit:
#                         grid_counts[row_idx, col] += 1
                        
#                         # Add the centroid and bounding box to the array
#                         xy_array.append([cx, cy, xmin, ymin, xmax, ymax])

#     return xy_array

# def calc_centroids(xy_array):
#     centroids_arr = []
#     centroids_and_box = []
#     for box in xy_array:
#         xmin, ymin, xmax, ymax = box[2], box[3], box[4], box[5]
#         cx, cy = box[0], box[1]
#         centroids_arr.append([cx, cy])
#         centroids_and_box.append([[cx, cy], [xmin, ymin, xmax, ymax]])
#     return centroids_arr, centroids_and_box

# def save_csv(camera_id, file_path, centroids_and_box):
#     flattened_data = [[cen_x, cen_y, xmin, ymin, xmax, ymax] for [cen_x, cen_y], [xmin, ymin, xmax, ymax] in centroids_and_box]
#     df = pd.DataFrame(flattened_data, columns=['cen_x', 'cen_y', 'xmin', 'ymin', 'xmax', 'ymax'])
#     df.to_csv(file_path + camera_id + ".csv", index=False)

# def grid_counting(camera_id):
#     # Load the image to get its dimensions
#     image_path = get_first_file(ROOT_DIR + "/" + camera_id)
#     image = cv2.imread(image_path)
#     image_height, image_width, _ = image.shape

#     # Define grid dimensions
#     num_rows = 10
#     num_cols = 10
#     x_step = image_width / num_cols
#     y_step = image_height / num_rows

#     # Initialize the grid count matrix
#     grid_counts = np.zeros((num_rows, num_cols))

#     return grid_counts, num_rows, num_cols, x_step, y_step

# def get_first_file(directory):
#     # List all files in the directory
#     files = os.listdir(directory)
    
#     # Get the first file
#     if len(files) == 0:
#         print("No files found in the directory.")
#         return None
    
#     first_file = files[0]
    
#     # Construct full file path
#     file_path = os.path.join(directory, first_file)
    
#     return file_path

# # List of camera IDs
# camera_ids = [
#     "AYER RAJAH EXPRESSWAY/4701"
# ]


# for camera_id in camera_ids:
#     # Get grid setup
#     grid_counts, num_rows, num_cols, x_step, y_step = grid_counting(camera_id)

#     limit = 200         # Set to -1 to ignore limit per grid
#     # Apply YOLO-NAS and get centroids
#     xy_array = apply_yolo_nas_l(camera_id, grid_counts, num_rows, num_cols, x_step, y_step, limit)

#     # Calculate centroids and save CSV
#     centroids_arr, centroids_and_box = calc_centroids(xy_array)
#     save_csv(camera_id.split("/")[1], "C:/Users/Jess/Desktop/School/FYP/CAM-R/models/experiments/lane_detection/centroidsv3/", centroids_and_box)
#     print("Done w/ " + camera_id)


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

pre_image_min = 100
pre_min_grid = 7
min_per_grid = 45
max_per_grid = 200 
total_image_process = 1500 # hard cap for the algorithm to stop running

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
                            
                    if (grid_check is True and len(grid_indexes) == 0) or (image_count > 1500):
                        break
        except Exception as e:
            print(e)
            print(grid_indexes)

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
    save_csv(camera_id.split("/")[1], "C:/Users/Jess/Desktop/School/FYP/CAM-R/models/experiments/lane_detection/centroids/", centroids_and_box)
    
    print(f"Done w/ {camera_id}. Total Centroids Collected: {total_centroid_count}. Images Processed: {image_count}")
