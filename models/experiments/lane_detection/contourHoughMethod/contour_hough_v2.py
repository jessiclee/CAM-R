# -*- coding: utf-8 -*-
"""
This method utilises the dominant road masking via bounding box scaling and overlapping.
If centroids does not land in the dominant mask, we do not consider the point, effectively removing road into our consideration
"""


import pandas as pd
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.neighbors import NearestNeighbors


##########################################MAIN PIPELINE################################################
def save_polygons_to_file(polygons, file_path):
    with open(file_path, 'w') as file:
        for poly in polygons:
            # Convert each polygon to a string with floating-point numbers
            poly_str = ' '.join(f"{x:.6f},{y:.6f}" for (x, y) in poly.squeeze())
            # Write the polygon to the file
            file.write(poly_str + '\n')

# Function to check if a point is inside any of the contours
def is_point_in_contours(point, contours):
    for contour in contours:
        # Convert point to int and tuple for pointPolygonTest
        point_tuple = tuple(map(int, point))
        if cv2.pointPolygonTest(contour, point_tuple, False) >= 0:
            return True
    return False

def resize_polygon(polygon, target_size):
    # Calculate the centroid of the polygon
    centroid = np.mean(polygon, axis=0)
    
    # Calculate the scaling factor based on target size
    x, y, w, h = cv2.boundingRect(polygon)
    polygon_width = w
    polygon_height = h
    
    width_scale = target_size[0] / polygon_width
    height_scale = target_size[1] / polygon_height
    scale = min(width_scale, height_scale)
    
    # Resize polygon
    resized_polygon = (polygon - centroid) * scale + centroid
    return resized_polygon

def pipeline(roadNum, is_hd):
  roadNum = str(roadNum)
  main_folder_dir = "C:/Users/Zhiyi/Desktop/FYP/newtraffic/"

  # Load binary image
  binary_image_path = main_folder_dir + "road_detection/" + roadNum + '.jpg'
  binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)

  # Find contours in the binary image
  contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  # get the centroid csv of the road
  df = pd.read_csv(main_folder_dir + roadNum +'.csv')
  df['ymax'] = df['ymax'].astype(int)
  # Read the centroid coordinates from CSV
  x = df['cen_x'].values
  y = df['ymax'].values
  data = np.array(list(zip(x, y)))

  # if the centroid is not within the contours of the dominant road mask, then remove centroid from data
  # Filter coordinates
  data = [coord for coord in data if is_point_in_contours(coord, contours)]

  # creating a black binary image with image dimensions (in our case, the HD is 1920 x 1080, while the low quality images are 320 x 240)
  image_width = 2000
  image_height = 2000

  if is_hd == False:
    image_width = 320
    image_height = 240

  binary_img = np.zeros((image_height, image_width), dtype=np.uint8)  # Define height and width of your image

  # Filtering of centroids
  filtered_data = []

  if is_hd == True:
    distance_threshold = 10 # Define the distance threshold
    nbrs = NearestNeighbors(n_neighbors=6, radius=distance_threshold).fit(data) # Create a NearestNeighbors object
    distances, indices = nbrs.radius_neighbors(data, radius=distance_threshold) # Find the neighbors
    filtered_data = [point for point, neighbors in zip(data, distances) if len(neighbors) > 4] # Filter out points that do not have more than 5 neighbors
    filtered_data = np.array(filtered_data)
  else:
    distance_threshold = 3.5 # Define the distance threshold
    nbrs = NearestNeighbors(n_neighbors=20, radius=distance_threshold).fit(data) # Create a NearestNeighbors object
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

  if is_hd == False:
    threshold_y = highest_y + 100
    line_width = 2
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Swap points if y1 > y2 to make calculations easier
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
            x2_extrapolated = x2 + (lowest_y - y2) * (x1 - x2) // (y1 - y2)
            y2_extrapolated = lowest_y

            # Case 1: If both points are below the threshold, draw from threshold to lowest_y
            if y1 >= threshold_y and y2 >= threshold_y:
                ratio = (threshold_y - y1) / (y2 - y1)
                new_x1 = int(x1 + ratio * (x2 - x1))
                new_y1 = threshold_y
                cv2.line(line_res, (new_x1, new_y1), (x2_extrapolated, y2_extrapolated), (255, 0, 0), line_width)

            # Case 2: If y1 is above the threshold but y2 is below, adjust the line
            elif y1 < threshold_y and y2 >= threshold_y:
                # Calculate intersection with threshold
                ratio = (threshold_y - y1) / (y2 - y1)
                new_x1 = int(x1 + ratio * (x2 - x1))
                new_y1 = threshold_y
                cv2.line(line_res, (new_x1, new_y1), (x2_extrapolated, y2_extrapolated), (255, 0, 0), line_width)

            # Case 3: If y2 is above the threshold but y1 is below, adjust the line
            elif y2 < threshold_y and y1 >= threshold_y:
                # Calculate intersection with threshold
                ratio = (threshold_y - y2) / (y1 - y2)
                new_x2 = int(x2 + ratio * (x1 - x2))
                new_y2 = threshold_y
                cv2.line(line_res, (x1, y1), (new_x2, new_y2), (255, 0, 0), line_width)

  ## FOR HD :< big duplication stuff idk why itts fucking up
  else:
    if lines is not None:
        for line in lines:
                x1,y1,x2,y2 = line[0]
                # Extrapolate the lines downward to the lowest centroid

                # Determine the lower point and extrapolate it to the lowest_y
                if y1 < y2:  # Extrapolate the lower point (y1) to lowest_y
                    x1 = x1 + (lowest_y - y1) * (x2 - x1) // (y2 - y1)
                    y1 = lowest_y
                elif y1 == y2:
                # Directly draw the vertical line
                    if y1 >= threshold_y:
                        # Draw the vertical line from threshold_y to lowest_y
                        cv2.line(line_res, (x1, threshold_y), (x1, lowest_y), (255, 0, 0), line_width)
                    continue
                else:
                    x2 = x2 + (lowest_y - y2) * (x1 - x2) // (y1 - y2)
                    y2 = lowest_y

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

  kernel = np.ones((3,3), np.uint8)
  if is_hd == True:
    line_res = cv2.erode(line_res, kernel, iterations=4)
    line_res = cv2.dilate(line_res, kernel, iterations=5)
  else:
    line_res = cv2.erode(line_res, kernel, iterations=2)
    line_res = cv2.dilate(line_res, kernel, iterations=2)

  # Find contours of the lanes
  gray = line_res
  _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
  contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#########################CHANGE DIR############################################
  road = cv2.imread(main_folder_dir + 'images/' + roadNum + '.jpg')

  min_area = 4000

  if is_hd == False:
    min_area = 200

  polygon_points = []
  for contour in contours:
      # Compute the area of the contour
      area = cv2.contourArea(contour)
      # Check if the area is greater than the minimum threshold
      if area > min_area:
          # Optionally, draw the rotated rectangle
          rect = cv2.minAreaRect(contour)
          box = cv2.boxPoints(rect)
          box_width = rect[1][0]
          box_height = rect[1][1]

          resized_box = resize_polygon(box, (50,200))
          resized_box = resized_box.astype(np.int32)
          polygon_points.append(resized_box)
          cv2.polylines(road, [resized_box], isClosed=True, color=(255, 0, 0), thickness=2)
  # Save to file
  save_polygons_to_file(polygon_points,  main_folder_dir + "v2result/autopolygons/" + roadNum + '.txt')

  #########################CHANGE DIR############################################
  if is_hd == False:
    cv2.imwrite( main_folder_dir + "v2result/low_road/" + roadNum + ".jpg", road)
  else:
    cv2.imwrite( main_folder_dir + "v2result/" + roadNum + ".jpg", road)
  print("saved " + roadNum + " results to folder!")

roads = [4701, 4705, 4706, 4707, 4708, 4709, 4710, 4712, 4714, 4716, 4799,
       2703, 2704, 2705, 2706, 2707, 2708, 1701, 1702, 1704, 1705, 1706,
       1707, 1709, 1711,  3702, 3705, 3793, 3795, 3796, 3797, 3704, 8701, 8702, 8704,
       5794, 5795, 5797, 6701, 6705, 6706, 6708, 6710, 6711, 6712, 6713,
       6714,  6715, 7797, 7798, 9701, 9702, 9703, 9704, 9705, 9706,
       1111, 1112, 7791, 7793, 7794, 7795, 7796]
low_roads = [1001, 1002, 1003, 1501, 1502, 1503, 1505, 1005, 1006, 1504]

for i in roads:
  pipeline(i, True)

for i in low_roads:
  pipeline(i, False)

