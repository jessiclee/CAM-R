import numpy as np
import pandas as pd
import cv2
import os
from sklearn.neighbors import NearestNeighbors

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

def find_dominant_roads(final_mask, width, height):

    # find contours 
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # create blank canvas for dominant mask
    dominant_mask = np.zeros((height, width, 3), dtype=np.uint8)

    min_area = 10000
    if width < 500:
        min_area = 400

    for contour in contours:
        # Compute the area of the contour
        area = cv2.contourArea(contour)
        # Check if the area is greater than the minimum threshold
        if area > min_area:
            # draw contour on the new canvas
            cv2.drawContours(dominant_mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    return dominant_mask

def get_camera_size_from_filename(camera_id):
    camera_id = int(camera_id)
    # Find the corresponding size based on camera ID
    if camera_id in camera_sizes['small']['cameras']:
        width = camera_sizes['small']['width']
        height = camera_sizes['small']['height']
    elif camera_id in camera_sizes['large']['cameras']:
        width = camera_sizes['large']['width']
        height = camera_sizes['large']['height']
    
    return height, width

def road_detect(roadNum):
    roadNum = str(roadNum)
    main_folder_dir = "C:/Users/Zhiyi/Desktop/FYP/newtraffic/"
    # get the centroid csv of the road
    df = pd.read_csv(main_folder_dir + roadNum +'.csv')

    # Read the centroid coordinates from CSV
    df = pd.read_csv(main_folder_dir + roadNum +'.csv')
    x = df['cen_x'].values
    y = df['cen_y'].values
    data = np.array(list(zip(x, y)))

    # Create a blank image with the same dimensions as the binary image
    height, width = get_camera_size_from_filename(roadNum)
    blank_image = np.zeros((height, width), dtype=np.uint8)

        # Filtering of centroids
    filtered_data = []

    if height == 1080:
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

    # Draw the centroids on the blank image
    for (x, y) in data:
        cv2.circle(blank_image, (int(x), int(y)), 3, (255), -1)  # Draw white circles
    # Apply dilation to the blank image
    kernel = np.ones((5,5), np.uint8)  # Adjust the kernel size as needed
    if height == 1080:
        dilated_image = cv2.dilate(blank_image, kernel, iterations=8)
        final_img = find_dominant_roads(dilated_image , width, height)
        cv2.imwrite("C:/Users/Zhiyi/Desktop/FYP/newtraffic/road_detection/" + roadNum + ".jpg", final_img)
    else:
        dilated_image = cv2.dilate(blank_image, kernel, iterations=3)
        final_img = find_dominant_roads(dilated_image , width, height)
        cv2.imwrite("C:/Users/Zhiyi/Desktop/FYP/newtraffic/road_detection/" + roadNum + ".jpg", final_img)


def main():
    # Extract camera IDs from the dictionary
    small_cameras = camera_sizes["small"]["cameras"]
    large_cameras = camera_sizes["large"]["cameras"]

    # Combine camera IDs into a single array
    all_camera_ids = small_cameras + large_cameras

    for i in all_camera_ids:
        print(i)
        road_detect(i)

main()