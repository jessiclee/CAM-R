""" # Example bounding box data structure (YOLO output)
# Each bbox contains (top_x, top_y, bottom_x, bottom_y, class)
bboxes = [
    {"x1": ..., "y1": ..., "x2": ..., "y2": ..., "class": "Car"},
    {"x1": ..., "y1": ..., "x2": ..., "y2": ..., "class": "Bus"},
    # More boxes
]

# Define vehicle properties
vehicle_properties = {
    "Car": {"length": 4.5, "safe_distance": 9},
    "Bus": {"length": 9, "safe_distance": 5},
    "Motorcycle": {"length": 1, "safe_distance": 2},
    "Truck": {"length": 8, "safe_distance": 5},
}

# Sort bboxes by increasing y1 (top of bbox)
bboxes = sorted(bboxes, key=lambda box: box["y1"])

curr_queue_list = []

for bbox in bboxes:
    vehicle_class = bbox["class"]
    vehicle_height = bbox["y2"] - bbox["y1"]  # Height of bbox
    length = vehicle_properties[vehicle_class]["length"]
    safe_distance = vehicle_properties[vehicle_class]["safe_distance"]
    
    # Calculate min_Pixel_dist
    min_Pixel_dist = vehicle_height * safe_distance / length
    
    # Calculate next vehicle point
    next_vehicle_point = bbox["y1"] + min_Pixel_dist
    
    # Add bbox to queue
    curr_queue_list.append(bbox)
    
    # Check the next bbox in the queue
    for next_bbox in curr_queue_list:
        if next_bbox["y2"] > next_vehicle_point:
            # Cut off the queue
            break """

####################################################################################################################
""" #faster method 0(N) >> O(N2)

# Example bounding box data structure (YOLO output)
# Each bbox contains (top_x, top_y, bottom_x, bottom_y, class)
bboxes = [
    {"x1": ..., "y1": ..., "x2": ..., "y2": ..., "class": "Car"},
    {"x1": ..., "y1": ..., "x2": ..., "y2": ..., "class": "Bus"},
    # More boxes
]

# Define vehicle properties
vehicle_properties = {
    "Car": {"length": 4.5, "safe_distance": 9},
    "Bus": {"length": 9, "safe_distance": 5},
    "Motorcycle": {"length": 1, "safe_distance": 2},
    "Truck": {"length": 8, "safe_distance": 5},
}

# Sort bboxes by increasing y1 (top of bbox)
bboxes = sorted(bboxes, key=lambda box: box["y1"])

# Sliding window approach with two pointers
curr_queue_list = []
start = 0  # Start pointer

for i, bbox in enumerate(bboxes):
    vehicle_class = bbox["class"]
    vehicle_height = bbox["y2"] - bbox["y1"]  # Height of bbox
    length = vehicle_properties[vehicle_class]["length"]
    safe_distance = vehicle_properties[vehicle_class]["safe_distance"]
    
    # Calculate min_Pixel_dist
    min_Pixel_dist = vehicle_height * safe_distance / length
    
    # Calculate next vehicle point
    next_vehicle_point = bbox["y1"] + min_Pixel_dist
    
    # Add current bbox to queue
    curr_queue_list.append(bbox)
    
    # Remove bounding boxes that are too far behind (i.e., outside the window)
    while curr_queue_list[start]["y2"] <= next_vehicle_point:
        start += 1  # Slide window forward
    
    # Process only the bounding boxes in the valid range (between start and i)
    for j in range(start, i + 1):
        next_bbox = curr_queue_list[j]
        if next_bbox["y2"] > next_vehicle_point:
            # Exit early when distance condition is violated
            break
 """

##############################################################################################################
# queue drawing by converting to pxiel coordinates
"""
    Convert YOLO format (center_x, center_y, width, height) to pixel coordinates for OpenCV.
    
    Args:
        normalized_bbox (list): YOLO formatted bbox [center_x, center_y, width, height].
        img_width (int): Width of the image in pixels.
        img_height (int): Height of the image in pixels.
    
    Returns:
        dict: Bounding box with pixel coordinates (x1, y1, x2, y2).
    """
"""
    Draws the bounding box and next_vehicle_point on the image.
    
    Args:
        image (numpy array): The image on which to draw.
        bbox (dict): The bounding box with coordinates and class.
        vehicle_class (str): The class of the vehicle.
        next_vehicle_point (float): The y-coordinate of the next vehicle point.
    """
""" import cv2
import cv2

def convert_yolo_to_bbox(normalized_bbox, img_width, img_height):
    
    class_id, cx, cy, w, h = normalized_bbox
    
    # Convert normalized coordinates to pixel values
    x1 = int((cx - w / 2) * img_width)
    y1 = int((cy - h / 2) * img_height)
    x2 = int((cx + w / 2) * img_width)
    y2 = int((cy + h / 2) * img_height)
    
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "class": class_id}

# Now you can use the pixel_bboxes in the process_bboxes() function to draw them

def draw_bboxes_on_image(image, bbox, vehicle_class, next_vehicle_point):
    
    # Get bounding box coordinates
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

    # Draw the bounding box in green
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Label the bounding box with the vehicle class
    cv2.putText(image, vehicle_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Draw the next_vehicle_point as a red line
    cv2.line(image, (x1, int(next_vehicle_point)), (x2, int(next_vehicle_point)), (0, 0, 255), 2)

    return image

# Sliding window and processing logic
def process_bboxes(bboxes, vehicle_properties, image):
    # Sort bboxes by increasing y1 (top of bbox)
    bboxes = sorted(bboxes, key=lambda box: box["y1"])

    # Sliding window approach with two pointers
    curr_queue_list = []
    start = 0  # Start pointer

    for i, bbox in enumerate(bboxes):
        vehicle_class = bbox["class"]
        vehicle_height = bbox["y2"] - bbox["y1"]  # Height of bbox
        length = vehicle_properties[vehicle_class]["length"]
        safe_distance = vehicle_properties[vehicle_class]["safe_distance"]

        # Calculate min_Pixel_dist
        min_Pixel_dist = vehicle_height * safe_distance / length

        # Calculate next vehicle point
        next_vehicle_point = bbox["y1"] + min_Pixel_dist

        # Add current bbox to queue
        curr_queue_list.append(bbox)

        # Remove bounding boxes that are too far behind (i.e., outside the window)
        while curr_queue_list[start]["y2"] <= next_vehicle_point:
            start += 1  # Slide window forward

        # Draw current bounding box and next_vehicle_point
        image = draw_bboxes_on_image(image, bbox, vehicle_class, next_vehicle_point)

        # Process only the bounding boxes in the valid range (between start and i)
        for j in range(start, i + 1):
            next_bbox = curr_queue_list[j]
            if next_bbox["y2"] > next_vehicle_point:
                # Exit early when distance condition is violated
                break

    return image

# Example usage
yolo_bboxes = [
    [3, 0.707299, 0.520190, 0.049422, 0.088083],
    [3, 0.629185, 0.136120, 0.016005, 0.029352],
    [3, 0.622406, 0.428366, 0.039438, 0.071028],
    [3, 0.622169, 0.101708, 0.016047, 0.025565],
    [3, 0.768263, 0.783495, 0.079057, 0.155898],
    [3, 0.596023, 0.276509, 0.027255, 0.058019],
]

# Assuming image size (replace with actual image dimensions)
img_width = 1280
img_height = 720

# Convert YOLO bounding boxes to pixel coordinates
pixel_bboxes = [convert_yolo_to_bbox(bbox, img_width, img_height) for bbox in yolo_bboxes]

# Bounding box properties (based on class 3 being some vehicle, like Car or Truck)
vehicle_properties = {
    3: {"length": 4.5, "safe_distance": 9},  # Example properties for class 3 (modify as needed)
}

# Assuming image size and loading an image
image = cv2.imread("path_to_your_image.jpg")
img_height, img_width = image.shape[:2]

# Convert YOLO bounding boxes to pixel coordinates
pixel_bboxes = [convert_yolo_to_bbox(bbox, img_width, img_height) for bbox in yolo_bboxes]

# Process and draw the bounding boxes on the image
output_image = process_bboxes(pixel_bboxes, vehicle_properties, image)

# Display and save the image
cv2.imshow("Output Image", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("output_image.jpg", output_image)
 """

##############################################################################################################
import cv2

def draw_all_bboxes(image, bbox, vehicle_class, next_vehicle_point_normalized):
    """
    Draws the bounding box and next_vehicle_point on the image, using normalized coordinates.
    
    Args:
        image (numpy array): The image on which to draw.
        bbox (dict): The bounding box with normalized coordinates (YOLO format).
        vehicle_class (str): The class of the vehicle.
        next_vehicle_point_normalized (float): The y-coordinate of the next vehicle point (normalized).
        img_width (int): Width of the image in pixels.
        img_height (int): Height of the image in pixels.
    """
    img_height, img_width = image.shape[:2]  # Get image dimensions

    # Get bounding box in normalized format
    cx, cy, w, h = bbox["cx"], bbox["cy"], bbox["w"], bbox["h"]
    
    # Convert normalized coordinates to pixel values for drawing
    x1 = int((cx - w / 2) * img_width)
    y1 = int((cy - h / 2) * img_height)
    x2 = int((cx + w / 2) * img_width)
    y2 = int((cy + h / 2) * img_height)
    
    # Convert normalized next_vehicle_point to pixel y-coordinate
    next_vehicle_point_pixel = int(next_vehicle_point_normalized * img_height)

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, str(vehicle_class), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Draw the next_vehicle_point as a horizontal red line
    cv2.line(image, (x1, next_vehicle_point_pixel), (x2, next_vehicle_point_pixel), (0, 0, 255), 2)

    #  save the output image
    return image

def draw_queue_bboxes(image, bbox, vehicle_class, next_vehicle_point_normalized):
    """
    Draws the bounding box and next_vehicle_point on the image, using normalized coordinates.
    
    Args:
        image (numpy array): The image on which to draw.
        bbox (dict): The bounding box with normalized coordinates (YOLO format).
        vehicle_class (str): The class of the vehicle.
        next_vehicle_point_normalized (float): The y-coordinate of the next vehicle point (normalized).
        img_width (int): Width of the image in pixels.
        img_height (int): Height of the image in pixels.
    """
    img_height, img_width = image.shape[:2]  # Get image dimensions

    # Get bounding box in normalized format
    cx, cy, w, h = bbox["cx"], bbox["cy"], bbox["w"], bbox["h"]
    
    # Convert normalized coordinates to pixel values for drawing
    x1 = int((cx - w / 2) * img_width)
    y1 = int((cy - h / 2) * img_height)
    x2 = int((cx + w / 2) * img_width)
    y2 = int((cy + h / 2) * img_height)
    
    # Convert normalized next_vehicle_point to pixel y-coordinate
    next_vehicle_point_pixel = int(next_vehicle_point_normalized * img_height)

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    print(vehicle_class)
    cv2.putText(image, str(vehicle_class), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Draw the next_vehicle_point as a horizontal red line
    cv2.line(image, (x1, next_vehicle_point_pixel), (x2, next_vehicle_point_pixel), (0, 0, 255), 2)

    #  save the output image
    cv2.imwrite("output_image.jpg", image)     

def process_bboxes_normalized(bboxes, vehicle_properties, image):
    """
    Process YOLO-style bounding boxes with normalized coordinates without converting them to pixel coordinates.
    
    Args:
        bboxes (list): List of YOLO-style bounding boxes with normalized coordinates.
        vehicle_properties (dict): Dictionary of vehicle properties (length, safe_distance).
        image (numpy array): The image on which to draw.
    
    Returns:
        numpy array: The image with bounding boxes and next_vehicle_point drawn.
    """
    
    # Sort bboxes by increasing y1 (normalized center_y - height/2)
    bboxes = sorted(bboxes, key=lambda box: box["cy"])

    curr_queue_list = []
    #start = 0 
    next_vehicle_point_normalized = 0
    output_image = 0

    for i, bbox in enumerate(bboxes):
        vehicle_class = bbox["class"]
        vehicle_height_normalized = bbox["h"] 
        ratio = vehicle_properties[vehicle_class]

        # Calculate min_Pixel_dist (normalized)
        min_normalized_dist = vehicle_height_normalized * ratio

        # Calculate next vehicle point (normalized)
        next_vehicle_point_normalized = (bbox["cy"] + bbox["h"] / 2) + min_normalized_dist

        # Add current bbox to queue
        curr_queue_list.append(bbox)

        # Draw current bounding box and next_vehicle_point
        output_image = draw_all_bboxes(image, bbox, vehicle_class, next_vehicle_point_normalized)

    for i, bbox in enumerate(bboxes):
        print(curr_queue_list)
        vehicle_class = bbox["class"]
        vehicle_height_normalized = bbox["h"] 
        ratio = vehicle_properties[vehicle_class]

        print(next_vehicle_point_normalized)
        print(bbox["cy"] - bbox["h"]/2)
        if next_vehicle_point_normalized > 0 and (bbox["cy"] - bbox["h"]/2 ) > next_vehicle_point_normalized:
            break

        # Calculate min_Pixel_dist (normalized)
        min_normalized_dist = vehicle_height_normalized * ratio

        # Calculate next vehicle point (normalized)
        next_vehicle_point_normalized = (bbox["cy"] + bbox["h"] / 2) + min_normalized_dist

        # Add current bbox to queue
        curr_queue_list.append(bbox)

        # Draw current bounding box and next_vehicle_point
        draw_queue_bboxes(output_image, bbox, vehicle_class, next_vehicle_point_normalized)

    """  
       # Remove bounding boxes that are too far behind (normalized)
        while start < len(curr_queue_list) and (curr_queue_list[start]["cy"] + curr_queue_list[start]["h"] / 2) <= next_vehicle_point_normalized:
            start += 1  # Slide window forward 
        
         # Process only the bounding boxes in the valid range (between start and i)
        for j in range(start, i + 1):
            next_bbox = curr_queue_list[j]
            if (next_bbox["cy"] + next_bbox["h"] / 2) > :
                # Exit early when distance condition is violated
                break 
    """

    return curr_queue_list

def read_yolo_bboxes(file_path):
    yolo_bboxes = []
    
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()  # Split the line into components
            if len(parts) == 5:  # Ensure there are 5 parts (class, cx, cy, w, h)
                vehicle_class = int(parts[0])  # Convert class to integer
                cx = float(parts[1])  # Convert cx to float
                cy = float(parts[2])  # Convert cy to float
                w = float(parts[3])  # Convert width to float
                h = float(parts[4])  # Convert height to float
                
                # Append the dictionary to the list
                yolo_bboxes.append({
                    "class": vehicle_class,
                    "cx": cx,
                    "cy": cy,
                    "w": w,
                    "h": h
                })
    
    return yolo_bboxes

# YOLO-style bounding boxes (normalized coordinates)
""" yolo_bboxes = [
    {"class": 3, "cx": 0.707299, "cy": 0.520190, "w": 0.049422, "h": 0.088083},
    {"class": 3, "cx": 0.629185, "cy": 0.136120, "w": 0.016005, "h": 0.029352},
    {"class": 3, "cx": 0.622406, "cy": 0.428366, "w": 0.039438, "h": 0.071028},
    {"class": 3, "cx": 0.622169, "cy": 0.101708, "w": 0.016047, "h": 0.025565},
    {"class": 3, "cx": 0.768263, "cy": 0.783495, "w": 0.079057, "h": 0.155898},
    {"class": 3, "cx": 0.596023, "cy": 0.276509, "w": 0.027255, "h": 0.058019},
]
 """
# Define vehicle properties 
vehicle_properties = {
    3: 3.0, 
    4: 2.0,
    6: 2.1,
    8: 1.7,
}

#Load labels
yolo_bboxes = read_yolo_bboxes('C:\\Users\\jesle\\Desktop\\fyp\\actual_data\\queue_length_test\\4704_01-06-2024_18-25-02.txt')

# Load the image
image = cv2.imread('C:\\Users\\jesle\\Desktop\\fyp\\actual_data\\queue_length_test\\4704_01-06-2024_18-25-02.jpg')

# Process the bounding boxes and return queue list
queue_list = process_bboxes_normalized(yolo_bboxes, vehicle_properties, image)
queue_length = len(queue_list)
print(f"queue_length: {queue_length}")

