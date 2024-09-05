import cv2
import numpy as np
import json

# Read the mask image (ensure it's a binary image)
mask = cv2.imread('C:\\Users\\Jess\\Desktop\\School\\FYP\\CAM-R\\models\\experiments\\lane_detection\\colabmask\\colabmask-4701-road.jpg', cv2.IMREAD_GRAYSCALE)


# Threshold the image to binary if needed
_, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# Find contours in the binary mask
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define a minimum area threshold to filter out tiny contours
min_area_threshold = 1000  # Adjust this threshold based on your needs

# Create a blank mask of the same size as the original mask
filtered_mask = np.zeros_like(binary_mask)

# Filter out tiny contours based on area and draw the remaining contours on the new mask
filtered_contours = []
polygon_data = []

for contour in contours:
    if cv2.contourArea(contour) >= min_area_threshold:
        # Draw the contour on the filtered mask
        cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Add the contour to the list of filtered contours
        filtered_contours.append(contour)
        
        # Store the polygon coordinates (x, y points) as tuples
        coordinates = [(int(point[0][0]), int(point[0][1])) for point in contour]
        polygon_data.append(coordinates)

# Save the filtered mask
cv2.imwrite('filtered_mask.jpg', filtered_mask)

# Save the polygon data to a JSON file
with open('polygon_data.json', 'w') as json_file:
    json.dump(polygon_data, json_file, indent=4)

print(f"Polygon data has been saved to 'polygon_data.json'.")

# Count the number of polygons (filtered contours)
num_polygons = len(filtered_contours)

print(f"Number of polygons (after filtering): {num_polygons}")


# import cv2
# import numpy as np

# # Load the mask image
# mask = cv2.imread('C:\\Users\\Jess\\Desktop\\School\\FYP\\CAM-R\\models\\experiments\\lane_detection\\colabmask\\colabmask-4701-road.jpg', cv2.IMREAD_GRAYSCALE)


# # Define the kernel size for morphological operations
# kernel = np.ones((10, 10), np.uint8)  # Adjust kernel size based on your need

# # Apply dilation to connect smaller polygons to larger ones
# dilated_mask = cv2.dilate(mask, kernel, iterations=2)

# # Apply morphological closing to fill gaps
# closed_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel)

# # Save the resulting mask
# cv2.imwrite('connected_polygons.jpg', closed_mask)

# # Optionally, display the result (if running locally)
# # cv2.imshow('Connected Polygons', closed_mask)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
