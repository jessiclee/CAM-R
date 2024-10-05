import cv2
import json
def draw_bbox_with_annotation(image, bbox, dist):
    """
    Draws the bounding box on the image and annotates the distance between the current 
    bbox's ymin and the previous bbox's ymax.
    
    Args:
        image (numpy array): The image to draw on.
        bbox (dict): The bounding box to be drawn, with keys 'x1', 'y1', 'x2', 'y2', and 'class'.
        dist (float): The distance between the ymin of the current bbox and the ymax of the previous bbox.
    """
    print('start')
    # Drawing the bounding box (
    x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Annotate the distance
    cv2.putText(image, f"Dist: {dist:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    print("stop")
    return image

def queue_length_top_to_bottom(bboxes, vehicle_properties, image):
    """
    Processes a list of bounding boxes in a single lane, groups them into queues, 
    and draws the bounding boxes on the image with annotated distances.
    
    Args:
        bboxes (list): List of bounding boxes in the format [cx, cy, x1, y1, x2, y2, class_id].
        image (numpy array): The image to draw on.
        vehicle_properties (dict): A dictionary with vehicle class_id as key and property (e.g., length ratio) as value.
    
    Returns:
        list: A list of queue lists, where each queue list contains the bounding boxes in that queue.
    """
    output_image = image

    # Sort the bounding boxes by their centroid's y-coordinate (cy)
    bboxes = sorted(bboxes, key=lambda box: box["y1"])

    print(bboxes)

    # Initialize the list to hold queue lists
    queue_lists = []

    # Start the first queue list with the first bounding box
    queue_list = [bboxes[0]]
    prev_ymax = bboxes[0]["y2"]  # Initial ymax from the first bbox
    output_image = draw_bbox_with_annotation(output_image, bboxes[0], 0)

    # Loop through remaining bounding boxes
    for i in range(1, len(bboxes)):
        bbox = bboxes[i]
        
        # Calculate the height of the current bounding box
        bbox_height = bbox["y2"] - bbox["y1"]
        
        # Get the vehicle property for this class of vehicle
        vehicle_class = bbox["class"]
        vehicle_property = vehicle_properties[vehicle_class]
        
        # Calculate the threshold value
        threshold_value = bbox_height * vehicle_property
        
        # Calculate the distance between the ymin of the current bbox and the ymax of the previous bbox
        dist = bbox["y1"] - prev_ymax
        
        # Draw the current bounding box and annotate the distance
        output_image = draw_bbox_with_annotation(output_image, bbox, dist)
        
        # Determine if the current bbox belongs in the same queue or a new one
        if dist < threshold_value:
            queue_list.append(bbox)
        else:
            queue_lists.append(queue_list)
            queue_list = [bbox]
            
        
        # Update prev_ymax to the current bbox's ymax
        prev_ymax = bbox["y2"]
        print(queue_lists)
    
        # Append the last queue list if it's not already added
    if queue_list:
        queue_lists.append(queue_list)

    longest_queue_index = max(range(len(queue_lists)), key=lambda i: len(queue_lists[i]))
    combined_queue = queue_lists[longest_queue_index]

    if len(combined_queue) <= 1:
        return combined_queue
    else:
        for i in range(longest_queue_index, 0, -1) :
            current  = queue_lists[i][0]
            ahead = queue_lists[i-1][-1]
            if current["y1"] - ahead["y2"] < another_threshold:
                combined_queue = queue_lists[i-1] + combined_queue
            else:
                break
        
        for i in range(longest_queue_index, len(queue_lists) - 1) :
            current  = queue_lists[i][-1]
            behind = queue_lists[i+1][0]
            if behind["y1"] - current["y2"] < another_threshold:
                combined_queue = combined_queue + queue_lists[i+1]
            else:
                break
        #  save the output image
    cv2.imwrite("output_image.jpg", output_image)  

    return combined_queue

def read_bboxes(file_path):
    lanes_dict = {}
    
    with open(file_path, 'r') as file:
        lanes_data = json.load(file)

    # Parse data
    for lane_id, lane_bboxes in lanes_data.items():
        lanes_dict[lane_id] = []

        for bbox in lane_bboxes:
            cx = bbox[0]
            cy = bbox[1]
            x1 = bbox[2]
            y1 = bbox[3]
            x2 = bbox[4]
            y2 = bbox[5]
            vehicle_class_id = bbox[6]
            
            # Append each bbox to the lane's list
            lanes_dict[lane_id].append({
                "cx": cx,
                "cy": cy,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "class": vehicle_class_id
            })
    
    return lanes_dict  # Return the dictionary of lanes

def compute_queue_lengths(lane_data, image, vehicle_properties, direction='up'):
    queue_lengths = {}
    
    # Process each lane's bounding boxes
    for lane_id, lane_bboxes in lane_data.items():
        # Determine the queue based on the direction
        if direction == "up":
            queue_list = queue_length_top_to_bottom(lane_bboxes, vehicle_properties, image)
        """ elif direction == "down":
            queue_list = queue_length_bottom_to_top(lane_bboxes, vehicle_properties, image)
        elif direction == "left":
            queue_list = queue_length_left_to_right(lane_bboxes, vehicle_properties, image)
        elif direction == "right":
            queue_list = queue_length_right_to_left(lane_bboxes, vehicle_properties, image)
        """
        # Calculate queue length for the current lane
        queue_length = len(queue_list)
        
        # Store lane_id and queue length in the dictionary
        queue_lengths[lane_id] = queue_length 
    
    return queue_lengths



# YOLO-style bounding boxes (lane_id :[ [centroid x,centroid y, x min, y min ,x max, y max, vehicle class id] for each vehicle], lane_direction??? )
"""
json_data = {
    "2": [[170, 175, 145, 150, 195, 201, 3], [123, 56, 107, 34, 140, 77, 1], [140, 107, 123, 92, 156, 122, 3]],
    "1": [[89, 83, 77, 69, 100, 96, 3]],
    "4": [[254, 46, 237, 33, 272, 60, 3]]
]
}
"""

#Define vehicle properties 
vehicle_properties = {
    3: 2.0, 
    1: 2.0,
    0: 2.1,
    4: 1.7,
}

#Load labels
lane_data = read_bboxes('C:\\Users\\jesle\\Desktop\\fyp\\actual_data\\queue_length_test\\now.txt')

# Load the image
image = cv2.imread('C:\\Users\\jesle\\Desktop\\fyp\\actual_data\\queue_length_test\\9702_01-06-2024_13-05-02.jpg')

#Output queue length for each lane
queue_lengths= compute_queue_lengths(lane_data, image, vehicle_properties)
print(queue_lengths)

