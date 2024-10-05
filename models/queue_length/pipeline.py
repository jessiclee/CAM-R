"""
MAIN PIPELINE CODE

SUMMARY:
this python file should integrate vehicle_detection.py --> vehicle_lane_assignment.py together
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
import cv2
import json
import tkinter as tk

##############################################################################
########################### VARIABLES ########################################
##############################################################################

#Define vehicle properties 
vehicle_properties = {
    3: 2.0, 
    1: 2.0,
    0: 2.1,
    4: 1.7,
}

# Remove all Logger notifications
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

# Setting up YOLO NAS model
yolo_nas_l = models.get('yolo_nas_s', num_classes=4, checkpoint_path="C:/Users/Zhiyi/Desktop/FYP/CAM-R/models/experiments/lane_detection/contourHoughMethod/19.pth")

camera_sizes = {
    "small": {
        "width": 320,
        "height": 240,
        "cameras": [1001, 1004, 1005, 1006, 1504, 1501, 1502, 1503, 1505, 1002, 1003]
    },
    "large": {
        "width": 1920,
        "height": 1080,
        "cameras": [4701, 4702, 4704, 4705, 4706, 4707, 4708, 4709, 4710, 4712, 4714, 
                    4716, 4798, 4799, 6716, 2703, 2704, 2705, 2706, 2707, 2708, 1701, 
                    1702, 1703, 1704, 1705, 1706, 1707, 1709, 1711, 1113, 3702, 3705, 
                    3793, 3795, 3796, 3797, 3798, 3704, 5798, 8701, 8702, 8704, 8706, 
                    5794, 5795, 5797, 5799, 6701, 6703, 6704, 6705, 6706, 6708, 6710, 
                    6711, 6712, 6713, 6714, 6715, 7797, 7798, 9701, 9702, 9703, 9704, 
                    9705, 9706, 1111, 1112, 7791, 7793, 7794, 7795, 7796, 4703, 4713, 
                    2701, 2702]
    }
}

##############################################################################
########################### FUNCTIONS ########################################
##############################################################################

def get_camera_size(camid):
    camera_id = int(camid)
    
    # Find the corresponding size based on camera ID
    if camera_id in camera_sizes['small']['cameras']:
        width = camera_sizes['small']['width']
        height = camera_sizes['small']['height']
    elif camera_id in camera_sizes['large']['cameras']:
        width = camera_sizes['large']['width']
        height = camera_sizes['large']['height']
    else:
        raise ValueError(f"Camera ID {camera_id} from file '{filename}' is not predefined.")
    
    return width, height

def load_lines_from_file(file_path):
    lines = []
    # Check if the file exists
    if not os.path.exists(file_path):
        # If the file doesn't exist, simply return an empty list or handle it as needed
        return lines

    with open(file_path, 'r') as file:
        for line in file:
            # Convert each line to a list of tuples, handling floating-point numbers
            points = [tuple(map(float, pt.split(','))) for pt in line.strip().split()]
            # Convert the list of tuples to a numpy array and ensure correct data type
            lines.append(np.array(points, dtype=np.float32))
    return lines

def get_vehicles_from_df(df):
    # get the centroid csv of the road
    x = df['cen_x'].values.astype(int)
    y = df['cen_y'].values.astype(int)
    xmin = df['xmin'].values.astype(int)
    ymin = df['ymin'].values.astype(int)
    xmax = df['xmax'].values.astype(int)
    ymax = df['ymax'].values.astype(int)
    classes = df['class'].values.astype(int)  # assuming classes should also be integers

    data = np.array(list(zip(x, y, xmin, ymin, xmax, ymax, classes)))
    return data

def point_to_segment_distance_with_projection(point, segment_start, segment_end):
    """
    Calculate the shortest distance from a point to a line segment and return the projection point.

    Args:
    - point: A tuple (x0, y0) representing the point.
    - segment_start: A tuple (x1, y1) representing the start of the line segment.
    - segment_end: A tuple (x2, y2) representing the end of the line segment.

    Returns:
    - distance: The shortest distance from the point to the segment.
    - projection: The coordinates (proj_x, proj_y) of the projection point on the segment.
    """
    x0, y0 = point
    x1, y1 = segment_start
    x2, y2 = segment_end

    # Vector from segment_start to segment_end
    dx, dy = x2 - x1, y2 - y1
    # If the segment is a point, return the distance to that point
    if dx == 0 and dy == 0:
        return np.hypot(x0 - x1, y0 - y1), (x1, y1)
    
    # Project point onto the line (but not beyond the segment ends)
    t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))  # Clamp t to the segment
    
    # Find the projection point on the segment
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    
    # Return the distance from the point to the projection and the projection point
    distance = np.hypot(x0 - proj_x, y0 - proj_y)
    return distance, (proj_x, proj_y)

def point_to_polyline_distance_with_projection(point, polyline):
    """
    Calculate the shortest distance from a point to a polyline and return the projection point.

    Args:
    - point: A tuple (x0, y0) representing the point.
    - polyline: A list of tuples representing the polyline coordinates.

    Returns:
    - min_distance: The shortest distance from the point to the polyline.
    - closest_projection: The projection coordinates (proj_x, proj_y) on the polyline.
    """
    min_distance = float('inf')
    closest_projection = None
    
    for i in range(len(polyline) - 1):
        segment_start = polyline[i]
        segment_end = polyline[i + 1]
        distance, projection = point_to_segment_distance_with_projection(point, segment_start, segment_end)
        if distance < min_distance:
            min_distance = distance
            closest_projection = projection
    
    return min_distance, closest_projection

def line_assignment_by_perpendicular_distance(centroid, lines):
    closest_line = None
    closest_distance = float('inf')
    proj_coord = None
    lane_id = 1
    isValid = True
    for line in lines:
        distance_from_lane, coord = point_to_polyline_distance_with_projection(centroid, line)
        if distance_from_lane < closest_distance:
            closest_distance = distance_from_lane
            closest_line = lane_id
            proj_coord = coord
        lane_id += 1
    
    if closest_distance > 100:
        isValid = False

    return closest_line, proj_coord, isValid

# Function to get screen size
def get_screen_size():
    root = tk.Tk()
    root.withdraw() 
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return (width, height)

def get_centroids_df(centroids_and_box, class_indx):
    # Flatten the array and create a DataFrame
    flattened_data = [[cen_x, cen_y, xmin, ymin, xmax, ymax] for [cen_x, cen_y], [xmin, ymin, xmax, ymax] in centroids_and_box]

    # Create the DataFrame with headers
    df = pd.DataFrame(flattened_data, columns=['cen_x', 'cen_y', 'xmin', 'ymin', 'xmax', 'ymax'])
    df["class"] = class_indx
    return df

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

def apply_yolo_nas_l(image_path):
    # List of accepted labels
    accepted_list = [1,2,3,4]  # Example labels that are accepted
    filename = image_path
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust based on your image file types
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
            # print(label)
            if label in accepted_list:
                # print(pred.bboxes_xyxy[index])
                bboxes.append(pred.bboxes_xyxy[index])
                class_indx.append(label)
                conf.append(pred.confidence.astype(float)[index])
            
        # Update the filtered image with filtered detections
        xy_array.append(np.array(bboxes))
        centroids_arr, centroids_and_box = calc_centroids(xy_array)
        df = get_centroids_df(centroids_and_box, class_indx)
        
        return df

def draw_bbox_with_annotation(image, bbox, dist):
    """
    Draws the bounding box on the image and annotates the distance between the current 
    bbox's ymin and the previous bbox's ymax.
    
    Args:
        image (numpy array): The image to draw on.
        bbox (dict): The bounding box to be drawn, with keys 'x1', 'y1', 'x2', 'y2', and 'class'.
        dist (float): The distance between the ymin of the current bbox and the ymax of the previous bbox.
    """
    # Drawing the bounding box (
    x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Annotate the distance
    cv2.putText(image, f"Dist: {dist:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return image

def queue_length_top_to_bottom(bboxes, vehicle_properties, image):
    """
    Processes a list of bounding boxes in a single lane, groups them into queues, 
    and draws the bounding boxes on the image with annotated distances.
    
    Args:
        bboxes (list): List of bounding boxes in the format [cx, cy, x1, y1, x2, y2, class_id].
        image (numpy array): The image to draw on.
        vehicle_properties (dict): A dictionary with vehicle class_id as key and property (e.g., length ratio) as value.
    
    Returns:
        list: A list of queue lists, where each queue list contains the bounding boxes in that queue.
    """
    output_image = image

    # Sort the bounding boxes by their centroid's y-coordinate (cy)
    bboxes = sorted(bboxes, key=lambda box: box["y1"])

    # Initialize the list to hold queue lists
    queue_lists = []

    # Start the first queue list with the first bounding box
    queue_list = [bboxes[0]]
    prev_ymax = bboxes[0]["y2"]  # Initial ymax from the first bbox
    output_image = draw_bbox_with_annotation(output_image, bboxes[0], 0)

    # Loop through remaining bounding boxes
    for i in range(1, len(bboxes)):
        bbox = bboxes[i]
        print(queue_list)
        # Calculate the height of the current bounding box
        bbox_height = bbox["y2"] - bbox["y1"]
        
        # Get the vehicle property for this class of vehicle
        vehicle_class = bbox["class"]
        vehicle_property = vehicle_properties[vehicle_class]
        
        # Calculate the threshold value
        threshold_value = bbox_height * vehicle_property
        
        # Calculate the distance between the ymin of the current bbox and the ymax of the previous bbox
        dist = bbox["y1"] - prev_ymax
        
        # Draw the current bounding box and annotate the distance
        output_image = draw_bbox_with_annotation(output_image, bbox, dist)
        
        # Determine if the current bbox belongs in the same queue or a new one
        if dist < threshold_value:
            queue_list.append(bbox)
        else:
            queue_lists.append(queue_list)
            queue_list = [bbox]
            
        
        # Update prev_ymax to the current bbox's ymax
        prev_ymax = bbox["y2"]
        print(queue_lists)
    
        # Append the last queue list if it's not already added
    if queue_list:
        queue_lists.append(queue_list)
    
    longest_queue_index = max(range(len(queue_lists)), key=lambda i: len(queue_lists[i]))
    combined_queue = queue_lists[longest_queue_index]

    
    if len(combined_queue) <= 1:
        return combined_queue
    else:
        for i in range(longest_queue_index, 0, -1) :
            current  = queue_lists[i][0]
            ahead = queue_lists[i-1][-1]
            if current["y1"] - ahead["y2"] < another_threshold:
                combined_queue = queue_lists[i-1] + combined_queue
            else:
                break
        
        for i in range(longest_queue_index, len(queue_lists) - 1) :
            current  = queue_lists[i][-1]
            behind = queue_lists[i+1][0]
            if behind["y1"] - current["y2"] < another_threshold:
                combined_queue = combined_queue + queue_lists[i+1]
            else:
                break
        #  save the output image
    cv2.imwrite("output_image.jpg", output_image)  

    return combined_queue

def read_bboxes(lanes_dict):
    # Parse data
    for lane_id, lane_bboxes in lanes_dict.items():
        lanes_dict[lane_id] = []

        for bbox in lane_bboxes:
            cx = bbox[0]
            cy = bbox[1]
            x1 = bbox[2]
            y1 = bbox[3]
            x2 = bbox[4]
            y2 = bbox[5]
            vehicle_class_id = bbox[6]
            
            # Append each bbox to the lane's list
            lanes_dict[lane_id].append({
                "cx": cx,
                "cy": cy,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "class": vehicle_class_id
            })
    
    return lanes_dict  # Return the dictionary of lanes

def compute_queue_lengths(lane_data, image, vehicle_properties, direction='up'):
    queue_lengths = {}
    
    # Process each lane's bounding boxes
    for lane_id, lane_bboxes in lane_data.items():
        # Determine the queue based on the direction
        if direction == "up":
            queue_list = queue_length_top_to_bottom(lane_bboxes, vehicle_properties, image)
        """ elif direction == "down":
            queue_list = queue_length_bottom_to_top(lane_bboxes, vehicle_properties, image)
        elif direction == "left":
            queue_list = queue_length_left_to_right(lane_bboxes, vehicle_properties, image)
        elif direction == "right":
            queue_list = queue_length_right_to_left(lane_bboxes, vehicle_properties, image)
        """
        # Calculate queue length for the current lane
        queue_length = len(queue_list)
        
        # Store lane_id and queue length in the dictionary
        queue_lengths[lane_id] = queue_length 
    
    return queue_lengths

##############################################################################
###########################DRIVER CODE########################################
##############################################################################

# input road
roadNum = input("Enter road ID: ")

# directory paths
main_folder_dir = "C:/Users/Zhiyi/Desktop/FYP/newtraffic/"

image_path = main_folder_dir + "centroidimages/" + roadNum + ".jpg"
test_image = cv2.imread(image_path)

lane_path = main_folder_dir + "v3result/manual2/lines/" + roadNum + ".txt"
lines = load_lines_from_file(lane_path)

width, height = get_camera_size(roadNum)

# draw lane lines on image
lane_id = 1
for line in lines:
    midpoint_x = int((line[0][0] + line[1][0]) / 2)
    midpoint_y = int((line[0][1] + line[1][1]) / 2)
    cv2.putText(test_image,str(lane_id), (midpoint_x,midpoint_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    line = line.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(test_image, [line], isClosed=False, color=(255, 0, 0), thickness=2)
    lane_id += 1

df = apply_yolo_nas_l(image_path)
centroids = get_vehicles_from_df(df)

result_dict = {}

# You can change the colors to ur liking 
color_dict = {
    1: (0, 0, 255), # red = truck
    2: (255, 0, 255), # pink = motorcycle
    3: (255,182,193), # grey ish = car
    4: (255, 255, 0) # neon blue = bus
}

for centroid in centroids:
    coord = (centroid[0], centroid[1])
    cv2.circle(test_image, coord, 5, (0, 200, 0), -1)
    # This prints out the coordinates of the centroids on the image
    cv2.putText(test_image,"(" + str(centroid[0]) +", " + str(centroid[1]) +")", (centroid[0] + 4, centroid[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    centroid_line_assignment, projection, isValid = line_assignment_by_perpendicular_distance(coord, lines)
    if isValid is True:
        cv2.circle(test_image, (int(projection[0]), int(projection[1])), 5, color_dict.get(centroid[6]), -1)
        grouping = result_dict.get(centroid_line_assignment)
        if grouping is None:
            grouping = [centroid]
        else:
            grouping.append(centroid)
        result_dict.update({centroid_line_assignment: grouping})
    else:
        cv2.circle(test_image, coord, 3, (0,0,0), -1)

# for key in result_dict:
#     result_dict[key] = [arr.tolist() for arr in result_dict[key]]
# json_data = json.dumps(result_dict)
# print(json_data)

#Load labels
lane_data = read_bboxes(result_dict)
#Output queue length for each lane
queue_lengths= compute_queue_lengths(lane_data, test_image, vehicle_properties)
print(queue_lengths)