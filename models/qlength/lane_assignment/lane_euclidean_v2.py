import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from PIL import Image
import pandas as pd
import torch
from super_gradients.training import models

# YOLO-NAS Model
yolo_nas_l = models.get('yolo_nas_s', num_classes=4, checkpoint_path="19.pth")

# Function to calculate centroids
def calc_centroids(xy_array):
    # print("xy array is " + str(len(xy_array)))
    centroids_arr = []
    centroids_and_box = []
    for image in xy_array:
        for box in image:
            xmin, ymin, xmax, ymax = box
            cx = int((xmin + xmax) / 2)
            cy = int((ymin + ymax) / 2)
            centroids_arr.append([cx, cy])
            centroids_and_box.append([[cx, cy], [xmin, ymin, xmax, ymax]])
    # print("centroids_and_box is  " + str(len(centroids_and_box)))
    return centroids_arr, centroids_and_box

# Function to apply YOLO-NAS to an image
def apply_yolo_nas_l(filename):
    accepted_list = [1, 2, 3, 4]
    
    if filename.endswith(".jpg") or filename.endswith(".png"):
        camera_id, extension = os.path.splitext(filename)
        camera_id = camera_id.split('_')[0]
        xy_array = []
        image_path = os.path.join(".", filename)
        image = Image.open(image_path)
        filtered_image = yolo_nas_l.predict(image, conf=0.5)
        bboxes = []
        class_indx = []
        conf = []
        pred = filtered_image.prediction
        labels = pred.labels.astype(int)
        
        for index, label in enumerate(labels):
            if label in accepted_list:
                bboxes.append(pred.bboxes_xyxy[index])
                class_indx.append(label)
                conf.append(pred.confidence.astype(float)[index])
        
        xy_array.append(np.array(bboxes))
        centroids_arr, centroids_and_box = calc_centroids(xy_array)
        centroid_bbox_list = []
        for i, ([cen_x, cen_y], [xmin, ymin, xmax, ymax]) in enumerate(centroids_and_box):
            centroid_bbox_list.append((cen_x, cen_y, xmin, ymin, xmax, ymax, class_indx[i]))
        return centroid_bbox_list

# Function to get key points from the lane contour
def get_key_points(contour, num_points=24):
    contour_length = cv2.arcLength(contour, False)
    key_points = []
    distance_between_points = contour_length / (num_points - 1)
    accumulated_length = 0
    prev_point = contour[0][0]
    key_points.append(prev_point)
    
    for i in range(1, len(contour)):
        point = contour[i][0]
        segment_length = np.linalg.norm(np.array(point) - np.array(prev_point))
        accumulated_length += segment_length
        if accumulated_length >= distance_between_points:
            key_points.append(point)
            accumulated_length = 0
            if len(key_points) >= num_points:
                break
        prev_point = point
    return key_points

def visualise(overlay_image, green_centroids, blue_centroids, centroids_outside_lane, lane_key_points, lane_assignments):
    # Plotting the result
    plt.figure(figsize=(10, 8))
    plt.imshow(overlay_image, cmap='gray')

    # Draw centroids inside lanes (green)
    for centroid in green_centroids:
        plt.scatter(centroid[0], centroid[1], color='green')

    # Plot centroids close to lanes (blue)
    for centroid in blue_centroids:
        plt.scatter(centroid[0], centroid[1], color='blue')

    # Plot centroids outside of any lane (red)
    for centroid in centroids_outside_lane:
        plt.scatter(centroid[0], centroid[1], color='red')

    # Visualize lane numbers
    for i, key_points in enumerate(lane_key_points):
        middle_point = key_points[len(key_points) // 2]
        plt.text(middle_point[0], middle_point[1], f"{i+1}", color='white', fontsize=8, fontweight='bold')

    plt.title('Lane Detection with Centroid Assignment and Key Point Reassignment')
    plt.axis('off')

    # Output lane assignments and reassignments
    for lane, assigned_centroids in lane_assignments.items():
        print(f"{lane}: {assigned_centroids}")
    print(f"Out of Lane (Red): {centroids_outside_lane}")

    # Display plot
    plt.show()
    
def save_json(lane_assignments, basename):
    # Save the dictionary with only cen_x, cen_y, and class to a JSON file
    lane_assignments_serializable = {
        lane: [[int(cen_x), int(cen_y), int(class_label)] for (cen_x, cen_y, xmin, ymin, xmax, ymax, class_label) in centroids]
        for lane, centroids in lane_assignments.items()
    }

    # Save to JSON file
    with open('accuracy_test/predictions/' + basename +'.json', 'w') as file:
        json.dump(lane_assignments_serializable, file)

def main_function(mask_path, predicting_image_path):
    # Load the provided mask image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Apply YOLO-NAS to the image to get centroids and bounding boxes
    centroids = apply_yolo_nas_l(predicting_image_path)
    predicting_image = cv2.imread(predicting_image_path, cv2.IMREAD_COLOR)

    # Convert the mask into binary (0 or 255 values)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the mask to identify lanes
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Overlay the mask onto the original image at low opacity
    overlay_image = predicting_image.copy()
    alpha = 0.3  # Transparency factor
    overlay_image[binary_mask == 255] = overlay_image[binary_mask == 255] * (1 - alpha) + 255 * alpha

    # Image dimensions
    img_height, img_width = binary_mask.shape

    # Threshold to assign proximity-based centroids (10% of image size)
    threshold_distance = 0.1 * min(img_width, img_height)

    # Get key points for each lane
    lane_key_points = []
    for contour in contours:
        key_points = get_key_points(contour, num_points=24)
        lane_key_points.append(key_points)

    # Assign centroids to lanes based on contours
    lane_assignments = {f"{i+1}": [] for i in range(len(contours))}
    centroids_outside_lane = []

    # Separate lists for visualizing green and blue centroids
    green_centroids = []
    blue_centroids = []

    for centroid in centroids:
        cen_x, cen_y, xmin, ymin, xmax, ymax, class_label = centroid
        assigned = False
        for i, contour in enumerate(contours):
            if cv2.pointPolygonTest(contour, (cen_x, cen_y), False) >= 0:
                lane_assignments[f"{i+1}"].append(centroid)
                green_centroids.append(centroid)
                assigned = True
                break
        if not assigned:
            centroids_outside_lane.append(centroid)

    # Reassign centroids that are close to the key points into lanes
    new_outside_lane = []
    centroid_distances = []

    for centroid in centroids_outside_lane:
        cen_x, cen_y, xmin, ymin, xmax, ymax, class_label = centroid
        min_distance = float('inf')
        closest_key_point = None
        closest_lane = None
        
        for i, key_points in enumerate(lane_key_points):
            for key_point in key_points:
                distance = np.linalg.norm(np.array([cen_x, cen_y]) - np.array(key_point))
                if distance < min_distance:
                    min_distance = distance
                    closest_key_point = key_point
                    closest_lane = f"{i+1}"
        
        if min_distance <= threshold_distance:
            lane_assignments[closest_lane].append(centroid)
            blue_centroids.append(centroid)
        else:
            new_outside_lane.append(centroid)

    centroids_outside_lane = new_outside_lane

    visualise(overlay_image, green_centroids, blue_centroids, centroids_outside_lane, lane_key_points, lane_assignments)
    save_json(lane_assignments, os.path.basename(mask_path).split('.')[0])

# Single usage
mask_path = '../../experiments/lane_detection/overlap_lane_masks/1706.jpg'
predicting_image_path = 'C:/Users/Jess/OneDrive - Singapore Management University/FYP/lane_assignment_testing/test_images/1706_01-07-2024_14-40-01.jpg'
main_function(mask_path, predicting_image_path)


# # Looping usage
# def loop_through_directories(mask_dir, predicting_image_dir):
#     # Get the list of files in both directories
#     mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.jpg') or f.endswith('.png')])
#     predicting_image_files = sorted([f for f in os.listdir(predicting_image_dir) if f.endswith('.jpg') or f.endswith('.png')])

#     # Check if both directories have the same number of files
#     if len(mask_files) != len(predicting_image_files):
#         print("Warning: The number of mask files and predicting image files do not match.")
    
#     # Loop through both lists, assuming they correspond to each other by name or order
#     for mask_file, predicting_image_file in zip(mask_files, predicting_image_files):
#         mask_path = os.path.join(mask_dir, mask_file)
#         predicting_image_path = os.path.join(predicting_image_dir, predicting_image_file)

#         # Call the main function for each pair of mask and predicting image
#         main_function(mask_path, predicting_image_path)

# # Example usage
# mask_dir = '../../experiments/lane_detection/overlap_lane_masks'
# predicting_image_dir = 'C:/Users/Jess/OneDrive - Singapore Management University/FYP/lane_assignment_testing/test_images'
# loop_through_directories(mask_dir, predicting_image_dir)