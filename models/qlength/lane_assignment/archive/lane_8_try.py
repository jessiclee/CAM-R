import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the provided mask image
mask_path = '../experiments/lane_detection/overlap_lane_masks/1701.jpg'
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Load the original image (same size as the mask)
predicting_image_path = '../experiments/lane_detection/test_images/CENTRAL EXPRESSWAY/1701/1701_19-05-2024_23-10-55.jpg'
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

# Increase the threshold to 15% for checking proximity
threshold_distance = 0.15 * min(img_width, img_height)

# Generate mock centroids
# centroids = [    (861, 336), (1271, 397), (1409, 335), (917, 240), (1840, 165),
#     (1799, 177), (1198, 170), (1230, 323), (1887, 184), (1262, 383),
#     (1071, 380), (1263, 383), (1161, 194), (1233, 186), (1713, 191),
#     (1853, 752), (1752, 195), (1843, 145), (985, 201), (1900, 175),
#     (1070, 380), (1206, 475), (801, 977), (725, 318), (1328, 340),
#     (999, 540), (1128, 221), (802, 589), (1288, 278), (1284, 225),
#     (1208, 175), (1402, 281), (1409, 269), (990, 233), (1407, 277),
#     (939, 569), (1026, 408), (897, 329), (470, 475), (1286, 223)]

centroids = [    (1271, 397)
# 
    ]

# Function to get 8 equally spaced points along the contour (lane)
def get_key_points(contour, num_points=8):
    contour_length = cv2.arcLength(contour, False)  # Get the length of the contour
    key_points = []
    distance_between_points = contour_length / (num_points - 1)  # Calculate the distance between key points
    accumulated_length = 0
    prev_point = contour[0][0]

    key_points.append(prev_point)  # Start with the first point
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

# Get key points for each lane
lane_key_points = []
for contour in contours:
    key_points = get_key_points(contour, num_points=8)
    lane_key_points.append(key_points)

# Assign centroids to lanes based on contours
lane_assignments = {f"Lane {i+1}": [] for i in range(len(contours))}
centroids_outside_lane = []
centroids_to_assign = []

for centroid in centroids:
    assigned = False
    for i, contour in enumerate(contours):
        if cv2.pointPolygonTest(contour, centroid, False) >= 0:  # Check if point is inside the contour
            lane_assignments[f"Lane {i+1}"].append(centroid)
            assigned = True
            break
    if not assigned:
        centroids_outside_lane.append(centroid)

# Reassign centroids that are close to the key points (blue) or leave them outside (red)
centroids_close_to_lane = []
centroid_distances = []  # Store distances for visualization

for centroid in centroids_outside_lane:
    min_distance = float('inf')
    closest_key_point = None
    
    # Find the minimum distance to any of the 8 key points in each lane
    for key_points in lane_key_points:
        for key_point in key_points:
            distance = np.linalg.norm(np.array(centroid) - np.array(key_point))
            print(f"Distance from centroid {centroid} to key point {key_point}: {distance:.2f}")
            if distance < min_distance:
                min_distance = distance
                closest_key_point = key_point
    
    # Store the distance for visualization
    centroid_distances.append((centroid, closest_key_point, min_distance))
    
    # If the minimum distance is within the threshold, assign the centroid to that lane
    if min_distance <= threshold_distance:
        centroids_close_to_lane.append(centroid)
        centroids_outside_lane.remove(centroid)
    else:
        print(f"Centroid {centroid} is too far from all key points.")

# Plotting the result
plt.figure(figsize=(10, 8))
plt.imshow(overlay_image, cmap='gray')

# Draw centroids in lanes (green), close to lanes (blue), and outside (red)
for i, (lane, assigned_centroids) in enumerate(lane_assignments.items()):
    for centroid in assigned_centroids:
        plt.scatter(centroid[0], centroid[1], color='green')

# Plot centroids close to lanes (blue)
for centroid in centroids_close_to_lane:
    plt.scatter(centroid[0], centroid[1], color='blue')

# Plot centroids outside of any lane (red)
for centroid in centroids_outside_lane:
    plt.scatter(centroid[0], centroid[1], color='red')

# Visualize key points for each lane (black 'x')
for key_points in lane_key_points:
    for key_point in key_points:
        plt.scatter(key_point[0], key_point[1], color='black', marker='x', s=100)

# Visualize lines connecting centroids and their closest key points
for centroid, closest_key_point, distance in centroid_distances:
    if closest_key_point is not None:
        plt.plot([centroid[0], closest_key_point[0]], [centroid[1], closest_key_point[1]], 'r--')

plt.title('Lane Detection with Centroid Assignment and Key Point Reassignment')
plt.axis('off')

# Output lane assignments and reassignments
for lane, assigned_centroids in lane_assignments.items():
    print(f"{lane}: {assigned_centroids}")
print(f"Close to Lane (Blue): {centroids_close_to_lane}")
print(f"Out of Lane (Red): {centroids_outside_lane}")

# Display plot
plt.show()