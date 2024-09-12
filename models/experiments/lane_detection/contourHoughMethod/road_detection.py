import numpy as np
import pandas as pd
import cv2
import os

# Define camera sizes based on predefined IDs
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

def get_camera_size_from_filename(filename):
    # Extract the numeric camera ID from the filename (e.g., "1001.csv" -> 1001)
    camera_id_str = os.path.splitext(filename)[0]  # Remove ".csv" extension
    camera_id = int(camera_id_str)  # Convert to integer
    
    # Find the corresponding size based on camera ID
    if camera_id in camera_sizes['small']['cameras']:
        width = camera_sizes['small']['width']
        height = camera_sizes['small']['height']
    elif camera_id in camera_sizes['large']['cameras']:
        width = camera_sizes['large']['width']
        height = camera_sizes['large']['height']
    else:
        raise ValueError(f"Camera ID {camera_id} from file '{filename}' is not predefined.")
    
    return width, height

def scale_bounding_boxes(df, scale):
    # Convert DataFrame to a list of centroids and bounding boxes
    centroids_and_boxes = df[['cen_x', 'cen_y', 'xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
    
    scaled_boxes = []
    for cx, cy, xmin, ymin, xmax, ymax in centroids_and_boxes:
        # Scale the coordinates relative to the center
        xmin = int(cx + scale * (xmin - cx))
        ymin = int(cy + scale * (ymin - cy))
        xmax = int(cx + scale * (xmax - cx))
        ymax = int(cy + scale * (ymax - cy))

        # Append scaled box coordinates
        scaled_boxes.append((xmin, ymin, xmax, ymax))
    
    return scaled_boxes

def create_overlap_mask(scaled_boxes, width, height, overlap_threshold):
    # Initialize the mask with zeros
    mask_count = np.zeros((height, width), dtype=np.uint8)
    
    for (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax) in scaled_boxes:
        # Ensure coordinates are within the image bounds
        scaled_xmin = max(scaled_xmin, 0)
        scaled_ymin = max(scaled_ymin, 0)
        scaled_xmax = min(scaled_xmax, width - 1)
        scaled_ymax = min(scaled_ymax, height - 1)
        
        # Increment the count for the area covered by the current bounding box
        mask_count[scaled_ymin:scaled_ymax, scaled_xmin:scaled_xmax] += 1
    
    # Apply threshold to get the final mask
    final_mask = (mask_count >= overlap_threshold).astype(np.uint8)
    
    return final_mask

def find_dominant_roads(final_mask, width, height):

    # find contours 
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(len(contours))
    # create blank canvas for dominant mask
    dominant_mask = np.zeros((height, width, 3), dtype=np.uint8)

    min_area = 50000
    if width < 500:
        min_area = 100

    for contour in contours:
        # Compute the area of the contour
        area = cv2.contourArea(contour)
        # Check if the area is greater than the minimum threshold
        if area > min_area:
            # draw contour on the new canvas
            cv2.drawContours(dominant_mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    return dominant_mask

def process_files(csv_folder_path, scales, overlap_thresholds):
    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        csv_file_path = os.path.join(csv_folder_path, csv_file)
        df = pd.read_csv(csv_file_path)
        
        # Get the camera size based on the file name
        width, height = get_camera_size_from_filename(csv_file)
        
        for scale in scales:
            for overlap_threshold in overlap_thresholds:
                # Scale the bounding boxes
                scaled_boxes = scale_bounding_boxes(df, scale)
                
                # Create the overlap mask based on predefined width and height
                mask = create_overlap_mask(scaled_boxes, width, height, overlap_threshold)
                
                # Apply dominant masking
                mask = find_dominant_roads(mask, width, height)
                # Create the output directory if it doesn't exist
                output_dir = f'C:/Users/Zhiyi/Desktop/FYP/newtraffic/road_detection/'
                os.makedirs(output_dir, exist_ok=True)
                
                # Define the output file path
                output_file = os.path.join(output_dir, f'{os.path.splitext(csv_file)[0]}.jpg')
                
                # Save the mask
                cv2.imwrite(output_file, mask)
                print(f"Saved mask to {output_file}")

# Example usage
csv_folder_path = 'C:/Users/Zhiyi/Desktop/FYP/CAM-R/models/experiments/lane_detection/centroidsv2/'  # Folder containing your CSV files

# Array of scales and overlap thresholds to test
# # scales = [0.3, 0.35, 0.4, 0.45, 0.5]  # Array of scales to test
scales = [0.5]

overlap_thresholds = [2]

process_files(csv_folder_path, scales, overlap_thresholds)
