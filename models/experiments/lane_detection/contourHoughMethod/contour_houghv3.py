# -*- coding: utf-8 -*-
"""
Contour Hough Method Version 3.0
SUMMARY:
This method utilises the centroids plot to draw Hough lines over the dense areas.
Generating lines to mark lanes instead of polygons and extrapolate the line.

DEPENDENCIES: pandas, os, numpy, opencv, sklearn
"""
##############################################################################
###########################  IMPORTS  ########################################
##############################################################################
import pandas as pd
import os
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors

##############################################################################
########################### VARIABLES ########################################
##############################################################################
roads = [4701, 4705, 4706, 4707, 4708, 4709, 4710, 4712, 4714, 4716, 4799,
        2703, 2704, 2705, 2706, 2707, 2708, 1701, 1702, 1704, 1705, 1706,
        1707, 1709, 1711,  3702, 3705, 3793, 3795, 3796, 3797, 3704, 8701, 8702, 8704,
        5794, 5795, 5797, 6701, 6705, 6706, 6708, 6710, 6711, 6712, 6713,
        6714,  6715, 7797, 7798, 9701, 9702, 9703, 9704, 9705, 9706,
        1111, 1112, 7791, 7793, 7794, 7795, 7796]
low_roads = [1001, 1002, 1003, 1501, 1502, 1503, 1505, 1005, 1006, 1504]

# main working directory
main_folder_dir = "C:/Users/Zhiyi/Desktop/FYP/newtraffic/"

##############################################################################
########################### FUNCTIONS ########################################
##############################################################################

# Function to save generated polylines to file
def save_lines_to_file(lines, file_path):
    with open(file_path, 'w') as file:
        for line in lines:
            # Convert each polygon to a string with floating-point numbers
            line_str = ' '.join(f"{x:.6f},{y:.6f}" for (x, y) in line)
            # Write the polygon to the file
            file.write(line_str + '\n')

# Function to get the center line of the lane polygon
def fit_center_line(box):
    
    # Find the midpoints of all side
    mid_side_1 = ((box[0][0] + box[1][0]) // 2, (box[0][1] + box[1][1]) // 2)
    mid_side_2 = ((box[1][0] + box[2][0]) // 2, (box[1][1] + box[2][1]) // 2)
    mid_side_3 = ((box[2][0] + box[3][0]) // 2, (box[2][1] + box[3][1]) // 2)
    mid_side_4 = ((box[3][0] + box[0][0]) // 2, (box[3][1] + box[0][1]) // 2)
    
    line_length_1 = np.sqrt((mid_side_1[0] - mid_side_3[0])**2 + (mid_side_1[1] - mid_side_3[1])**2)
    line_length_2 = np.sqrt((mid_side_2[0] - mid_side_4[0])**2 + (mid_side_2[1] - mid_side_4[1])**2)
    if line_length_1 >= line_length_2:
        return mid_side_1[0], mid_side_1[1], mid_side_3[0], mid_side_3[1]
    return mid_side_2[0], mid_side_2[1], mid_side_4[0], mid_side_4[1]

# Function to extend center line to border
def extend_line_to_borders(x1, y1, x2, y2, width, height):
    # Handle the case where the line is vertical (avoid division by zero)
    if x1 == x2:
        # Vertical line: Extend to the top and bottom borders
        y_top = 0
        y_bottom = height
        return (x1, y_top), (x2, y_bottom)
    
    # Calculate the slope and intercept
    m = (y2 - y1) / (x2 - x1)  # Slope
    b = y1 - m * x1            # Intercept
    
    # Find intersections with the four borders
    # Left border (x = 0)
    y_at_left = b
    # Right border (x = width)
    y_at_right = m * width + b
    # Top border (y = 0)
    if m != 0:
        x_at_top = -b / m  # Avoid division by zero if m == 0
    else:
        x_at_top = float('inf')  # Impossible value for horizontal line
    # Bottom border (y = height)
    if m != 0:
        x_at_bottom = (height - b) / m
    else:
        x_at_bottom = float('inf')  # Impossible value for horizontal line

    # Collect valid points that lie within the image boundaries
    points = []
    
    if 0 <= y_at_left <= height:
        points.append((0, int(y_at_left)))  # Left border
    if 0 <= y_at_right <= height:
        points.append((width, int(y_at_right)))  # Right border
    if 0 <= x_at_top <= width:
        points.append((int(x_at_top), 0))  # Top border
    if 0 <= x_at_bottom <= width:
        points.append((int(x_at_bottom), height))  # Bottom border

    # We expect exactly two valid intersection points
    if len(points) == 2:
        return points[0], points[1]
    else:
        raise ValueError("Unexpected number of intersections found!")

# Function to extrapolate lines to y threshold
def extrapolate_lines(x1, y1, x2, y2, line_res, threshold_y, line_width):
    # Case 1: If both points are below the threshold, draw the line as is
    if y1 >= threshold_y and y2 >= threshold_y:
        cv2.line(line_res, (x1, y1), (x2, y2), (255, 0, 0), line_width)

    # Case 2: If y1 is above the threshold but y2 is below, adjust the line
    elif y1 < threshold_y and y2 >= threshold_y:
        # Calculate intersection with threshold
        ratio = (threshold_y - y1) / (y2 - y1)
        new_x1 = int(x1 + ratio * (x2 - x1))
        new_y1 = threshold_y
        cv2.line(line_res, (new_x1, new_y1), (x2, y2), (255, 0, 0), line_width)

    # Case 3: If y2 is above the threshold but y1 is below, adjust the line
    elif y2 < threshold_y and y1 >= threshold_y:
        # Calculate intersection with threshold
        ratio = (threshold_y - y2) / (y1 - y2)
        new_x2 = int(x2 + ratio * (x1 - x2))
        new_y2 = threshold_y
        cv2.line(line_res, (x1, y1), (new_x2, new_y2), (255, 0, 0), line_width)

# Driver Code Function
def pipeline(roadNum, is_hd, main_dir):
    roadNum = str(roadNum)
    main_folder_dir = main_dir

    # get the centroid csv of the road
    df = pd.read_csv(main_folder_dir + roadNum +'.csv')
    df['ymax'] = df['ymax'].astype(int)
    x = df['cen_x'].values
    y = df['ymax'].values
    data = np.array(list(zip(x, y)))

    # creating a black binary image with image dimensions 
    # (in our case, the HD is 1920 x 1080, while the low quality images are 320 x 240)
    image_width = 2000
    image_height = 2000
    if is_hd == False:
        image_width = 350
        image_height = 250

    # Create a binary image canvas to plot out the centroids
    binary_img = np.zeros((image_height, image_width), dtype=np.uint8)  # Define height and width of your image

    # Filtering of centroids via nearest neighbour
    filtered_data = []
    distance_threshold = 3.5
    num_neighbors = 20

    if is_hd == True:
        distance_threshold = 10
        num_neighbors = 6

    nbrs = NearestNeighbors(n_neighbors=num_neighbors, radius=distance_threshold).fit(data) # Create a NearestNeighbors object
    distances, indices = nbrs.radius_neighbors(data, radius=distance_threshold) # Find the neighbors
    filtered_data = [point for point, neighbors in zip(data, distances) if len(neighbors) > 4] # Filter out points that do not have more than 5 neighbors
    filtered_data = np.array(filtered_data)

    # Plot the filtered centroids into an image
    for xi, yi in filtered_data:
        binary_img[int(yi), int(xi)] = 255

    # Enhance the image contrast
    binary_img = cv2.equalizeHist(binary_img)
    kernel = np.ones((3,3), np.uint8)

    if is_hd == True:
        binary_img = cv2.dilate(binary_img, kernel, iterations=3)
    else:
        binary_img = cv2.dilate(binary_img, kernel, iterations=1)

    # Generation of Hough Lines
    rho = 1
    theta = np.pi/180
    threshold = 60
    minLineLength = 100
    maxLineGap = 15

    if is_hd == False:
        rho = 1
        theta = np.pi/180
        threshold = 50
        minLineLength = 50
        maxLineGap = 20

    lines = cv2.HoughLinesP(image=binary_img,rho=rho,theta=theta, threshold=threshold,minLineLength=minLineLength,maxLineGap=maxLineGap)
    if(lines is not None):
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(binary_img,(x1,y1),(x2,y2),(255, 0, 0 ),2)

    # Extrapolate and shrink the lines
    line_res = np.zeros((image_height, image_width), dtype=np.uint8)
    highest_y = np.min(y)
    lowest_y = np.max(y)
    threshold_y = highest_y + 400
    line_width = 4

    if is_hd is False:
        threshold_y = highest_y + 100
        line_width = 2

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Swap points if y1 > y2
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1

            # Handle vertical lines (y1 == y2)
            if y1 == y2:
                # Directly draw the vertical line
                if y1 >= threshold_y:
                    # Draw the vertical line from threshold_y to lowest_y
                    cv2.line(line_res, (x1, threshold_y), (x1, lowest_y), (255, 0, 0), line_width)
                continue  # Skip further calculations for vertical lines

            # Extrapolate line to lowest_y
            x2 = x2 + (lowest_y - y2) * (x1 - x2) // (y1 - y2)
            y2 = lowest_y
            extrapolate_lines(x1, y1, x2, y2, line_res, threshold_y, line_width)

    kernel = np.ones((3,3), np.uint8)
    erosion_iter = 2
    dilation_iter = 2
    if is_hd == True:
        erosion_iter = 4
        dilation_iter = 5
    line_res = cv2.erode(line_res, kernel, iterations=erosion_iter)
    line_res = cv2.dilate(line_res, kernel, iterations=dilation_iter)


    # Find contours of the lanes lines generated by Hough Transform
    gray = line_res
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # retrieve sample road image
    road = cv2.imread(main_folder_dir + 'images/' + roadNum + '.jpg')

    # determine the minimum area to be considered as a lane
    min_area = 3000
    if is_hd == False:
        min_area = 200

    line_points = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # Check if the area is greater than the minimum threshold
        if area > min_area:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            vx, vy, x, y = fit_center_line(box)
            if vx is not None and vy is not None:
                # Extend the line to the image borders
                (x1, y1), (x2, y2) = extend_line_to_borders(vx, vy, x, y, image_width, image_height)
                # Draw the line on the image
                cv2.line(road, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red line
                line_points.append([(x1, y1), (x2, y2)])

    # saving lines to relevant folders
    save_lines_to_file(line_points,  main_folder_dir + "v3result/autolines/" + roadNum + '.txt')

    # saving image result to relevant folders
    if is_hd == False:
        cv2.imwrite( main_folder_dir + "v3result/low_road/" + roadNum + ".jpg", road)
    else:
        cv2.imwrite( main_folder_dir + "v3result/" + roadNum + ".jpg", road)
    
    print("saved " + roadNum + " results to folder!")

##############################################################################
###########################DRIVER CODE########################################
##############################################################################

for i in roads:
    pipeline(i, True, main_folder_dir)

for i in low_roads:
    pipeline(i, False, main_folder_dir)

