import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

# Load the provided mask image
mask_path = '../../experiments/lane_detection/overlap_lane_masks/1701.jpg'
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Load the original image (same size as the mask)
predicting_image_path = '../../experiments/lane_detection/test_images/CENTRAL EXPRESSWAY/1701/1701_19-05-2024_23-10-55.jpg'
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

# Generate mock centroids
centroids = [
    (861, 336), (1271, 397), (1409, 335), (917, 240), (1840, 165),
    (1799, 177), (1198, 170), (1230, 323), (1887, 184), (1262, 383),
    (1071, 380), (1263, 383), (1161, 194), (1233, 186), (1713, 191),
    (1853, 752), (1752, 195), (1843, 145), (985, 201), (1900, 175),
    (1070, 380), (1206, 475), (801, 977), (725, 318), (1328, 340),
    (999, 540), (1128, 221), (802, 589), (1288, 278), (1284, 225),
    (1208, 175), (1402, 281), (1409, 269), (990, 233), (1407, 277),
    (939, 569), (1026, 408), (897, 329), (470, 475), (1286, 223)
]

# Function to get 8 equally spaced points along the contour (lane)
def get_key_points(contour, num_points=16):
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
    key_points = get_key_points(contour, num_points=16)
    lane_key_points.append(key_points)

# Function to calculate the projection of a point onto a line segment (vector projection)
def project_point_onto_segment(p, a, b):
    ap = p - a
    ab = b - a
    ab_norm = ab / np.linalg.norm(ab)
    projection = a + np.dot(ap, ab_norm) * ab_norm
    return projection

# Function to check if the projection falls within the segment
def is_point_on_segment(p, a, b):
    return np.all((p >= np.minimum(a, b)) & (p <= np.maximum(a, b)))

# Assign centroids to lanes based on contours
lane_assignments = {f"Lane {i+1}": [] for i in range(len(contours))}
centroids_outside_lane = []
centroids_to_assign = []

# Separate lists for visualizing green and blue centroids
green_centroids = []
blue_centroids = []

# List to store projection lines for visualization
projection_lines = []

for centroid in centroids:
    assigned = False
    for i, contour in enumerate(contours):
        if cv2.pointPolygonTest(contour, centroid, False) >= 0:  # Check if point is inside the contour
            lane_assignments[f"Lane {i+1}"].append(centroid)
            green_centroids.append(centroid)  # These will be plotted green
            assigned = True
            break
    if not assigned:
        centroids_outside_lane.append(centroid)

# Reassign centroids based on vector projection and proximity check
new_outside_lane = []  # Temporary list to store centroids that remain outside

for centroid in centroids_outside_lane:
    closest_projection = None
    closest_distance = float('inf')
    closest_lane = None

    # Iterate through all key points in each lane to find the projection
    for i, key_points in enumerate(lane_key_points):
        for j in range(len(key_points) - 1):
            a = np.array(key_points[j])
            b = np.array(key_points[j + 1])
            p = np.array(centroid)
            
            # Project the point onto the line segment
            projection = project_point_onto_segment(p, a, b)
            
            # Check if the projection falls within the segment
            if is_point_on_segment(projection, a, b):
                # Calculate the distance from the point to the projection
                distance = np.linalg.norm(p - projection)
                
                # Keep track of the closest projection
                if distance < closest_distance:
                    closest_projection = projection
                    closest_distance = distance
                    closest_lane = f"Lane {i+1}"

    # If the closest projection is within the threshold, assign the centroid to the closest lane
    if closest_projection is not None and closest_distance <= threshold_distance:
        lane_assignments[closest_lane].append(centroid)
        blue_centroids.append(centroid)  # Keep these as blue
        # Store the projection line for visualization
        projection_lines.append((centroid, closest_projection))
    else:
        new_outside_lane.append(centroid)  # Store centroids that are still too far

# Update centroids_outside_lane after checking all centroids
centroids_outside_lane = new_outside_lane

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

# Visualize key points for each lane (black 'x')
for key_points in lane_key_points:
    for key_point in key_points:
        plt.scatter(key_point[0], key_point[1], color='black', marker='x', s=100)

# Visualize lines connecting centroids to their projections
for centroid, projection in projection_lines:
    plt.plot([centroid[0], projection[0]], [centroid[1], projection[1]], 'r--')

plt.title('Lane Detection with Vector Projection, Centroid Assignment, and Visualization')
plt.axis('off')

# Output lane assignments and reassignments
for lane, assigned_centroids in lane_assignments.items():
    print(f"{lane}: {assigned_centroids}")
print(f"Out of Lane (Red): {centroids_outside_lane}")

# Save the dictionary as a JSON file
lane_assignments_serializable = {lane: [list(centroid) for centroid in centroids] 
                                 for lane, centroids in lane_assignments.items()}
with open('lane_assignments.json', 'w') as file:
    json.dump(lane_assignments_serializable, file)

# Display plot
plt.show()
