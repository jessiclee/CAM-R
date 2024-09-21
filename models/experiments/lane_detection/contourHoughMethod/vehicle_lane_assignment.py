"""
Vehicle Lane Assignment

SUMMARY: 
Given a centroid, get the perpendicular distance then assign it to the lane that it is the closest to.
Centroid is generated using yolo_nas_centroids_generator.py
DEPENDENCIES: numpy, opencv, os, pandas, tkinter
"""

##############################################################################
###########################  IMPORTS  ########################################
##############################################################################
import cv2
import numpy as np
import os
import pandas as pd
import tkinter as tk

##############################################################################
########################### VARIABLES ########################################
##############################################################################
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

def get_vehicles_from_csv(file_path):
    # get the centroid csv of the road
    df = pd.read_csv(file_path)
    x = df['cen_x'].values
    y = df['cen_y'].values
    data = np.array(list(zip(x, y)))
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
    for line in lines:
        distance_from_lane, coord = point_to_polyline_distance_with_projection(centroid, line)
        if distance_from_lane < closest_distance:
            closest_distance = distance_from_lane
            closest_line = line
            proj_coord = coord
    return closest_line, proj_coord

# Function to get screen size
def get_screen_size():
    root = tk.Tk()
    root.withdraw() 
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return (width, height)

def save_line_assignment(line, bboxes):
    pass

##############################################################################
###########################DRIVER CODE########################################
##############################################################################
# input road
roadNum = input("Enter road ID: ")

# directory paths
main_folder_dir = "C:/Users/Zhiyi/Desktop/FYP/newtraffic/"
image_path = main_folder_dir + "centroidimages/" + roadNum + "/" + roadNum +".jpg"
lane_path = main_folder_dir + "v3result/manual/lines/" + roadNum + ".txt"
centroid_path = main_folder_dir + "centroidimages/" + roadNum + ".csv"
# get the saved polygons from the image and draw the polygons on a black image
lines = load_lines_from_file(lane_path)
test_image = cv2.imread(image_path)
width, height = get_camera_size(roadNum)

# draw lane lines on image
for line in lines:
    line = line.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(test_image, [line], isClosed=False, color=(255, 0, 0), thickness=2)

centroids = get_vehicles_from_csv(centroid_path)
for centroid in centroids:
    cv2.circle(test_image, centroid, 5, (0, 200, 0), -1)
    centroid_line_assignment, projection = line_assignment_by_perpendicular_distance(centroid, lines)
    cv2.circle(test_image, (int(projection[0]), int(projection[1])), 5, (0, 0, 255), -1)


cv2.namedWindow("lines", cv2.WINDOW_NORMAL)
screen_width, screen_height = get_screen_size()
cv2.resizeWindow("lines", screen_width, screen_height)
cv2.imshow("lines", test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()