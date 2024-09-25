import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original image (same size as the mask)
mask_path = '../experiments/lane_detection/overlap_lane_masks/1701.jpg'
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# # Load the original image (same size as the mask)
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

# Generate mock centroids
centroids = [    (861, 336), (1271, 397), (1409, 335), (917, 240), (1840, 165),
    (1799, 177), (1198, 170), (1230, 323), (1887, 184), (1262, 383),
    (1071, 380), (1263, 383), (1161, 194), (1233, 186), (1713, 191),
    (1853, 752), (1752, 195), (1843, 145), (985, 201), (1900, 175),
    (1070, 380), (1206, 475), (801, 977), (725, 318), (1328, 340),
    (999, 540), (1128, 221), (802, 589), (1288, 278), (1284, 225),
    (1208, 175), (1402, 281), (1409, 269), (990, 233), (1407, 277),
    (939, 569), (1026, 408), (897, 329), (470, 475), (1286, 223)]

# Function to calculate the vector projection of w onto v
def vector_projection(w, v):
    return (np.dot(w, v) / np.dot(v, v)) * v

# Assign centroids to lanes based on contours
lane_assignments = {f"Lane {i+1}": [] for i in range(len(contours))}
centroids_outside_lane = []
centroids_close_to_lane = []

# 1% of the image width and height for distance threshold
img_height, img_width = binary_mask.shape
threshold_distance = 0.1 * min(img_width, img_height)

for centroid in centroids:
    assigned = False
    min_distance = float('inf')
    
    # Step 1: Check if the centroid is inside any lane
    for i, contour in enumerate(contours):
        if cv2.pointPolygonTest(contour, centroid, False) >= 0:  # Inside the lane
            lane_assignments[f"Lane {i+1}"].append(centroid)
            assigned = True
            break
    
    # Step 2: If not inside any lane, check proximity using vector projection
    if not assigned:
        for i, contour in enumerate(contours):
            # Get two points along the contour to form a direction vector for the lane
            p1 = contour[0][0]
            p2 = contour[-1][0]
            
            # Vector representing the lane's direction
            v = np.array(p2) - np.array(p1)
            
            # Vector from centroid to the first point on the lane
            w = np.array(centroid) - np.array(p1)
            
            # Project the centroid's vector onto the lane's vector
            proj_w_on_v = vector_projection(w, v)
            
            # Calculate the projected point on the lane
            projected_point = np.array(p1) + proj_w_on_v
            
            # Calculate the Euclidean distance from the centroid to the projected point
            distance_to_lane = np.linalg.norm(np.array(centroid) - projected_point)
            
            # If within the threshold, mark as close to the lane
            if distance_to_lane <= threshold_distance:
                centroids_close_to_lane.append(centroid)
                assigned = True
                break
        
        # Step 3: If no proximity was detected, mark as outside the lane
        if not assigned:
            centroids_outside_lane.append(centroid)

# Plotting the result
plt.figure(figsize=(10, 8))
plt.imshow(overlay_image, cmap='gray')

# Draw centroids on the lanes (green for centroids inside lanes)
for i, (lane, assigned_centroids) in enumerate(lane_assignments.items()):
    for centroid in assigned_centroids:
        plt.scatter(centroid[0], centroid[1], color='green')

# Plot centroids close to lanes (blue)
for centroid in centroids_close_to_lane:
    plt.scatter(centroid[0], centroid[1], color='blue')

# Plot centroids outside of any lane in red
for centroid in centroids_outside_lane:
    plt.scatter(centroid[0], centroid[1], color='red')

plt.title('Lane Detection with Centroid Projection-based Assignment')
plt.axis('off')

# Output lane assignments
for lane, assigned_centroids in lane_assignments.items():
    print(f"{lane}: {assigned_centroids}")
print(f"Close to Lane (Blue): {centroids_close_to_lane}")
print(f"Out of Lane (Red): {centroids_outside_lane}")

# Display plot
plt.show()

