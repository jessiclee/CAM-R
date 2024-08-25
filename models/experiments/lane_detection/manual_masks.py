########################################################
# DEFINE YOUR OWN ROI CODE
# 1. Left click to connect or start the polygon
# 2. Right click to close the polygon
# 3. Repeat until ROI polygons are all drawn
########################################################

import cv2
import os
import numpy as np

# Initialize global variables
drawing = False  # True if mouse is pressed
all_polygons = []  # List to store all polygons
current_polygon = []  # Current polygon points
current_image = None

# Mouse callback function
def draw_polygons(event, x, y, flags, param):
    global drawing, current_polygon, current_image, all_polygons

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_polygon.append((x, y))
        if len(current_polygon) > 1:
            cv2.line(current_image, current_polygon[-2], current_polygon[-1], (0, 255, 0), 2)
        cv2.imshow("Image", current_image)

    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = False
        if len(current_polygon) > 2:
            cv2.line(current_image, current_polygon[-1], current_polygon[0], (0, 255, 0), 2)
            cv2.imshow("Image", current_image)
            all_polygons.append(current_polygon)
            current_polygon = []  # Reset current polygon for the next one

# Load the image
path = '4714-bg.jpg'        #have your image ready here
image = cv2.imread(path)
imS = cv2.resize(image, (960, 540))   
current_image = imS.copy()
cv2.imshow("Image", current_image)

# Set the mouse callback function to capture the polygons
cv2.setMouseCallback("Image", draw_polygons)

# Wait for the user to finish drawing polygons
cv2.waitKey(0)

# Create a mask from the selected polygons
mask = np.zeros_like(imS)

for polygon in all_polygons:
    if len(polygon) > 2:
        polygon_np = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [polygon_np], (255, 255, 255))

# Apply the mask to the image
masked_image = cv2.bitwise_and(imS, mask)

# Show the masked image
cv2.imshow("Masked Image", masked_image)
cv2.waitKey(0)

# Save the masked image if needed
cv2.imwrite('roads/mask-'+ os.path.basename(path) , masked_image)

cv2.destroyAllWindows()
