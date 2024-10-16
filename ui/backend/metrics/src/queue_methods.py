import os
from PIL import Image
import numpy as np
import pandas as pd
import torch
import cv2
import json
import math


def load_lanes(lanes):
    lines = []

    # Split the decoded string into lines
    for line in lanes.strip().splitlines():
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

def draw_queue(image, queue):
    for bbox in queue:
        x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
    return image

def queue_length_top_to_bottom(bboxes, vehicle_properties, image, queue_threshold, img_name):
    output_image = image

    # Sort the bounding boxes by their centroid's y-coordinate (cy)
    bboxes = sorted(bboxes, key=lambda box: box["y1"])

    # Initialize the list to hold queue lists
    queue_lists = []

    # Start the first queue list with the first bounding box
    queue_list = [bboxes[0]]
    prev_y = bboxes[0]["cy"] 
    prev_x = bboxes[0]["cx"] 
    output_image = draw_bbox_with_annotation(output_image, bboxes[0], 0)

    # Loop through remaining bounding boxes
    for i in range(1, len(bboxes)):
        bbox = bboxes[i]
        # Calculate the height of the current bounding box
        bbox_height = bbox["y2"] - bbox["y1"]
        
        # Get the vehicle property for this class of vehicle
        vehicle_class = bbox["class"]
        vehicle_property = vehicle_properties[vehicle_class]
        
        # Calculate the threshold value
        threshold_value = bbox_height * vehicle_property
        
        # Calculate the distance between the current bbox and the previous bbox
        dist = math.sqrt((bbox["cx"] - prev_x) ** 2 + (bbox["cy"] - prev_y) ** 2)
        
        # Draw the current bounding box and annotate the distance
        output_image = draw_bbox_with_annotation(output_image, bbox, dist)
        
        # Determine if the current bbox belongs in the same queue or a new one
        if dist < threshold_value:
            queue_list.append(bbox)
        else:
            queue_lists.append(queue_list)
            queue_list = [bbox]
            
        # Update prev_ymax to the current bbox's ymax
        prev_y = bbox["cy"]
        prev_x = bbox["cx"]
    
        # Append the last queue list if it's not already added
    if queue_list:
        queue_lists.append(queue_list)

    
    longest_queue_index = max(range(len(queue_lists)), key=lambda i: len(queue_lists[i]))
    combined_queue = queue_lists[longest_queue_index]

    if len(combined_queue) <= 1:
        output_image = draw_queue(output_image, combined_queue)
        # cv2.imwrite(f"./results/output_image_{img_name}.jpg", output_image) 
        # print("image saved at!!!", f"./results/output_image_{img_name}.jpg")
        return combined_queue
    else:
        for i in range(longest_queue_index, 0, -1) :
            current  = queue_lists[i][0]
            ahead = queue_lists[i-1][-1]
            dist = math.sqrt((current["cx"] - ahead["cx"]) ** 2 + (current["cy"] - ahead["cy"]) ** 2)
            if dist < queue_threshold:
                combined_queue = queue_lists[i-1] + combined_queue
            else:
                break
        
        for i in range(longest_queue_index, len(queue_lists) - 1) :
            current  = queue_lists[i][-1]
            behind = queue_lists[i+1][0]
            dist = math.sqrt((behind["cx"] - current["cx"]) ** 2 + (behind["cy"] - current["cy"]) ** 2)
            if dist < queue_threshold:
                combined_queue = combined_queue + queue_lists[i+1]
            else:
                break
     
    output_image = draw_queue(output_image, combined_queue)
    # cv2.imwrite(f"./results/output_image_{img_name}.jpg", output_image) 
    # print("image saved at!!!", f"./results/output_image_{img_name}.jpg")
    return combined_queue

def queue_length_bottom_to_top(bboxes, vehicle_properties, image, queue_threshold, img_name):
    output_image = image

    # Sort the bounding boxes by their centroid's y-coordinate (cy)
    bboxes = sorted(bboxes, key=lambda box: box["y1"], reverse=True)

    # Initialize the list to hold queue lists
    queue_lists = []

    # Start the first queue list with the first bounding box
    queue_list = [bboxes[0]]
    prev_y = bboxes[0]["cy"] 
    prev_x = bboxes[0]["cx"] 
    output_image = draw_bbox_with_annotation(output_image, bboxes[0], 0)

    # Loop through remaining bounding boxes
    for i in range(1, len(bboxes)):
        bbox = bboxes[i]
        # Calculate the height of the current bounding box
        bbox_height = bbox["y2"] - bbox["y1"]
        
        # Get the vehicle property for this class of vehicle
        vehicle_class = bbox["class"]
        vehicle_property = vehicle_properties[vehicle_class]
        
        # Calculate the threshold value
        threshold_value = bbox_height * vehicle_property
        
        # Calculate the distance between the current bbox and the previous bbox
        dist = math.sqrt((bbox["cx"] - prev_x) ** 2 + (bbox["cy"] - prev_y) ** 2)
        
        # Draw the current bounding box and annotate the distance
        output_image = draw_bbox_with_annotation(output_image, bbox, dist)
        
        # Determine if the current bbox belongs in the same queue or a new one
        if dist < threshold_value:
            queue_list.append(bbox)
        else:
            queue_lists.append(queue_list)
            queue_list = [bbox]
            
        # Update prev_ymax to the current bbox's ymax
        prev_y = bbox["cy"]
        prev_x = bbox["cx"]
    
        # Append the last queue list if it's not already added
    if queue_list:
        queue_lists.append(queue_list)

    
    longest_queue_index = max(range(len(queue_lists)), key=lambda i: len(queue_lists[i]))
    combined_queue = queue_lists[longest_queue_index]

    if len(combined_queue) <= 1:
        output_image = draw_queue(output_image, combined_queue)
        # cv2.imwrite(f"./results/output_image_{img_name}.jpg", output_image) 
        # print("image saved at!!!", f"./results/output_image_{img_name}.jpg")
        return combined_queue
    else:
        for i in range(longest_queue_index, 0, -1) :
            current  = queue_lists[i][0]
            ahead = queue_lists[i-1][-1]
            dist = math.sqrt((current["cx"] - ahead["cx"]) ** 2 + (current["cy"] - ahead["cy"]) ** 2)
            if dist < queue_threshold:
                combined_queue = queue_lists[i-1] + combined_queue
            else:
                break
        
        for i in range(longest_queue_index, len(queue_lists) - 1) :
            current  = queue_lists[i][-1]
            behind = queue_lists[i+1][0]
            dist = math.sqrt((behind["cx"] - current["cx"]) ** 2 + (behind["cy"] - current["cy"]) ** 2)
            if dist < queue_threshold:
                combined_queue = combined_queue + queue_lists[i+1]
            else:
                break
    
    output_image = draw_queue(output_image, combined_queue)
    # cv2.imwrite(f"./results/output_image_{img_name}.jpg", output_image) 
    # print("image saved at!!!", f"./results/output_image_{img_name}.jpg")
    return combined_queue

def queue_length_left_to_right(bboxes, vehicle_properties, image, queue_threshold, img_name):
    output_image = image

    # Sort the bounding boxes by their centroid's y-coordinate (cy)
    bboxes = sorted(bboxes, key=lambda box: box["x1"])

    # Initialize the list to hold queue lists
    queue_lists = []

    # Start the first queue list with the first bounding box
    queue_list = [bboxes[0]]
    prev_y = bboxes[0]["cy"] 
    prev_x = bboxes[0]["cx"] 
    output_image = draw_bbox_with_annotation(output_image, bboxes[0], 0)

    # Loop through remaining bounding boxes
    for i in range(1, len(bboxes)):
        bbox = bboxes[i]
        # Calculate the height of the current bounding box
        bbox_height = bbox["y2"] - bbox["y1"]
        
        # Get the vehicle property for this class of vehicle
        vehicle_class = bbox["class"]
        vehicle_property = vehicle_properties[vehicle_class]
        
        # Calculate the threshold value
        threshold_value = bbox_height * vehicle_property
        
        # Calculate the distance between the current bbox and the previous bbox
        dist = math.sqrt((bbox["cx"] - prev_x) ** 2 + (bbox["cy"] - prev_y) ** 2)
        
        # Draw the current bounding box and annotate the distance
        output_image = draw_bbox_with_annotation(output_image, bbox, dist)
        
        # Determine if the current bbox belongs in the same queue or a new one
        if dist < threshold_value:
            queue_list.append(bbox)
        else:
            queue_lists.append(queue_list)
            queue_list = [bbox]
            
        # Update prev_ymax to the current bbox's ymax
        prev_y = bbox["cy"]
        prev_x = bbox["cx"]
    
        # Append the last queue list if it's not already added
    if queue_list:
        queue_lists.append(queue_list)

    
    longest_queue_index = max(range(len(queue_lists)), key=lambda i: len(queue_lists[i]))
    combined_queue = queue_lists[longest_queue_index]

    if len(combined_queue) <= 1:
        output_image = draw_queue(output_image, combined_queue)
        # cv2.imwrite(f"./results/output_image_{img_name}.jpg", output_image) 
        # print("image saved at!!!", f"./results/output_image_{img_name}.jpg")
        return combined_queue
    else:
        for i in range(longest_queue_index, 0, -1) :
            current  = queue_lists[i][0]
            ahead = queue_lists[i-1][-1]
            dist = math.sqrt((current["cx"] - ahead["cx"]) ** 2 + (current["cy"] - ahead["cy"]) ** 2)
            if dist < queue_threshold:
                combined_queue = queue_lists[i-1] + combined_queue
            else:
                break
        
        for i in range(longest_queue_index, len(queue_lists) - 1) :
            current  = queue_lists[i][-1]
            behind = queue_lists[i+1][0]
            dist = math.sqrt((behind["cx"] - current["cx"]) ** 2 + (behind["cy"] - current["cy"]) ** 2)
            if dist < queue_threshold:
                combined_queue = combined_queue + queue_lists[i+1]
            else:
                break
    
    output_image = draw_queue(output_image, combined_queue)
    # cv2.imwrite(f"./results/output_image_{img_name}.jpg", output_image) 
    # print("image saved at!!!", f"./results/output_image_{img_name}.jpg")
    return combined_queue

def queue_length_right_to_left(bboxes, vehicle_properties, image, queue_threshold, img_name):
    output_image = image

    # Sort the bounding boxes by their centroid's y-coordinate (cy)
    bboxes = sorted(bboxes, key=lambda box: box["x1"], reverse=True)

    # Initialize the list to hold queue lists
    queue_lists = []

    # Start the first queue list with the first bounding box
    queue_list = [bboxes[0]]
    prev_y = bboxes[0]["cy"] 
    prev_x = bboxes[0]["cx"] 
    output_image = draw_bbox_with_annotation(output_image, bboxes[0], 0)

    # Loop through remaining bounding boxes
    for i in range(1, len(bboxes)):
        bbox = bboxes[i]
        # Calculate the height of the current bounding box
        bbox_height = bbox["y2"] - bbox["y1"]
        
        # Get the vehicle property for this class of vehicle
        vehicle_class = bbox["class"]
        vehicle_property = vehicle_properties[vehicle_class]
        
        # Calculate the threshold value
        threshold_value = bbox_height * vehicle_property
        
        # Calculate the distance between the current bbox and the previous bbox
        dist = math.sqrt((bbox["cx"] - prev_x) ** 2 + (bbox["cy"] - prev_y) ** 2)
        
        # Draw the current bounding box and annotate the distance
        output_image = draw_bbox_with_annotation(output_image, bbox, dist)
        
        # Determine if the current bbox belongs in the same queue or a new one
        if dist < threshold_value:
            queue_list.append(bbox)
        else:
            queue_lists.append(queue_list)
            queue_list = [bbox]
            
        # Update prev_ymax to the current bbox's ymax
        prev_y = bbox["cy"]
        prev_x = bbox["cx"]
    
        # Append the last queue list if it's not already added
    if queue_list:
        queue_lists.append(queue_list)

    
    longest_queue_index = max(range(len(queue_lists)), key=lambda i: len(queue_lists[i]))
    combined_queue = queue_lists[longest_queue_index]

    if len(combined_queue) <= 1:
        output_image = draw_queue(output_image, combined_queue)
        # cv2.imwrite(f"./results/output_image_{img_name}.jpg", output_image) 
        # print("image saved at!!!", f"./results/output_image_{img_name}.jpg")
        return combined_queue
    else:
        for i in range(longest_queue_index, 0, -1) :
            current  = queue_lists[i][0]
            ahead = queue_lists[i-1][-1]
            dist = math.sqrt((current["cx"] - ahead["cx"]) ** 2 + (current["cy"] - ahead["cy"]) ** 2)
            if dist < queue_threshold:
                combined_queue = queue_lists[i-1] + combined_queue
            else:
                break
        
        for i in range(longest_queue_index, len(queue_lists) - 1) :
            current  = queue_lists[i][-1]
            behind = queue_lists[i+1][0]
            dist = math.sqrt((behind["cx"] - current["cx"]) ** 2 + (behind["cy"] - current["cy"]) ** 2)
            if dist < queue_threshold:
                combined_queue = combined_queue + queue_lists[i+1]
            else:
                break
    
    output_image = draw_queue(output_image, combined_queue)
    # cv2.imwrite(f"./results/output_image_{img_name}.jpg", output_image) 
    # print("image saved at!!!", f"./results/output_image_{img_name}.jpg")
    return combined_queue

def queue_length_right_to_left(bboxes, vehicle_properties, image, queue_threshold, img_name):
    output_image = image

    # Sort the bounding boxes by their centroid's y-coordinate (cy)
    bboxes = sorted(bboxes, key=lambda box: box["x1"], reverse=True)

    # Initialize the list to hold queue lists
    queue_lists = []

    # Start the first queue list with the first bounding box
    queue_list = [bboxes[0]]
    prev_y = bboxes[0]["cy"] 
    prev_x = bboxes[0]["cx"] 
    output_image = draw_bbox_with_annotation(output_image, bboxes[0], 0)

    # Loop through remaining bounding boxes
    for i in range(1, len(bboxes)):
        bbox = bboxes[i]
        # Calculate the height of the current bounding box
        bbox_height = bbox["y2"] - bbox["y1"]
        
        # Get the vehicle property for this class of vehicle
        vehicle_class = bbox["class"]
        vehicle_property = vehicle_properties[vehicle_class]
        
        # Calculate the threshold value
        threshold_value = bbox_height * vehicle_property
        
        # Calculate the distance between the current bbox and the previous bbox
        dist = math.sqrt((bbox["cx"] - prev_x) ** 2 + (bbox["cy"] - prev_y) ** 2)
        
        # Draw the current bounding box and annotate the distance
        output_image = draw_bbox_with_annotation(output_image, bbox, dist)
        
        # Determine if the current bbox belongs in the same queue or a new one
        if dist < threshold_value:
            queue_list.append(bbox)
        else:
            queue_lists.append(queue_list)
            queue_list = [bbox]
            
        # Update prev_ymax to the current bbox's ymax
        prev_y = bbox["cy"]
        prev_x = bbox["cx"]
    
        # Append the last queue list if it's not already added
    if queue_list:
        queue_lists.append(queue_list)

    
    longest_queue_index = max(range(len(queue_lists)), key=lambda i: len(queue_lists[i]))
    combined_queue = queue_lists[longest_queue_index]

    if len(combined_queue) <= 1:
        output_image = draw_queue(output_image, combined_queue)
        # cv2.imwrite(f"./results/output_image_{img_name}.jpg", output_image) 
        # print("image saved at!!!", f"./results/output_image_{img_name}.jpg")
        return combined_queue
    else:
        for i in range(longest_queue_index, 0, -1) :
            current  = queue_lists[i][0]
            ahead = queue_lists[i-1][-1]
            dist = math.sqrt((current["cx"] - ahead["cx"]) ** 2 + (current["cy"] - ahead["cy"]) ** 2)
            if dist < queue_threshold:
                combined_queue = queue_lists[i-1] + combined_queue
            else:
                break
        
        for i in range(longest_queue_index, len(queue_lists) - 1) :
            current  = queue_lists[i][-1]
            behind = queue_lists[i+1][0]
            dist = math.sqrt((behind["cx"] - current["cx"]) ** 2 + (behind["cy"] - current["cy"]) ** 2)
            if dist < queue_threshold:
                combined_queue = combined_queue + queue_lists[i+1]
            else:
                break
    
    output_image = draw_queue(output_image, combined_queue)
    # cv2.imwrite(f"./predict/output_image_{img_name}.jpg", output_image) 
    # print("image saved at!!!", f"./results/output_image_{img_name}.jpg")
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

def compute_queue_lengths(lane_data, image, vehicle_properties, queue_threshold, direction, img_name):
    
    queue_lengths = {}
    # print(f"FUNCTION ENTERED FOR: {img_name}")
    
    # Process each lane's bounding boxes
    for lane_id, lane_bboxes in lane_data.items():
        queue_list = []
        # print("dir, lane", direction[lane_id-1], lane_id)
        # dir = direction[lane_id-1]
        dir = "up"
        # Determine the queue based on the direction
        if dir == "up":
            queue_list = queue_length_top_to_bottom(lane_bboxes, vehicle_properties, image, queue_threshold, (img_name +"----"+ str(lane_id)))
            # print('queue_list')
        elif direction == "down":
            queue_list = queue_length_bottom_to_top(lane_bboxes, vehicle_properties, image, queue_threshold)
        elif direction == "left":
            queue_list = queue_length_left_to_right(lane_bboxes, vehicle_properties, image, queue_threshold)
        elif direction == "right":
            queue_list = queue_length_right_to_left(lane_bboxes, vehicle_properties, image, queue_threshold)
       
        # Store lane_id and queue length in the dictionary
        queue_lengths[lane_id] = len(queue_list) 
        # cv2.imwrite(f"./predict/output_image_{img_name}.jpg", image) 
    
    return queue_lengths