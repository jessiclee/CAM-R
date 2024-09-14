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
    camera_id_str = os.path.splitext(filename)[0]
    camera_id = int(camera_id_str)
    
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
    centroids_and_boxes = df[['cen_x', 'cen_y', 'xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
    
    scaled_boxes = []
    for cx, cy, xmin, ymin, xmax, ymax in centroids_and_boxes:
        xmin = int(cx + scale * (xmin - cx))
        ymin = int(cy + scale * (ymin - cy))
        xmax = int(cx + scale * (xmax - cx))
        ymax = int(cy + scale * (ymax - cy))

        scaled_boxes.append((xmin, ymin, xmax, ymax))
    
    return scaled_boxes

def create_overlap_mask(scaled_boxes, width, height, overlap_threshold):
    mask_count = np.zeros((height, width), dtype=np.uint8)
    
    for (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax) in scaled_boxes:
        scaled_xmin = max(scaled_xmin, 0)
        scaled_ymin = max(scaled_ymin, 0)
        scaled_xmax = min(scaled_xmax, width - 1)
        scaled_ymax = min(scaled_ymax, height - 1)
        
        mask_count[scaled_ymin:scaled_ymax, scaled_xmin:scaled_xmax] += 1
    
    final_mask = (mask_count >= overlap_threshold).astype(np.uint8)
    
    return final_mask

def draw_and_save(image, output_file, original_width, original_height, mode="delete"):
    """
    Generic function to draw either curves (deletion) or lines (addition) on an image and save it.
    
    Parameters:
    - image: The image to draw on.
    - output_file: The file path where the image will be saved.
    - original_width: The original width of the image before any scaling.
    - original_height: The original height of the image before any scaling.
    - mode: A string that specifies the drawing mode ("delete" or "add").
    """
    original_image = image.copy()
    temp_image = image.copy()
    changes_made = False  # Flag to track if changes were made
    drawn_shapes = []  # List to store drawn shapes (curves or lines) and their thickness

    # Check if image should be scaled
    if original_width == 1920 and original_height == 1080:
        scale_factor = 0.5
    elif original_width == 320 and original_height == 240:
        scale_factor = 1.5
    else:
        scale_factor = 1.0
    
    if scale_factor != 1.0:
        image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
        temp_image = cv2.resize(temp_image, (0, 0), fx=scale_factor, fy=scale_factor)
    
    if mode == "delete":
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
    
    points = []
    thickness = 5 if mode == "delete" else 2  # Set default thickness

    def redraw_image():
        """
        Redraws the image with all the stored curves or lines.
        """
        nonlocal image, temp_image, mask
        image = original_image.copy()
        if scale_factor != 1.0:
            image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
        
        if mode == "delete":
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

        for shape, thickness in drawn_shapes:
            if mode == "delete":
                cv2.polylines(image, [shape], isClosed=False, color=(255, 0, 0), thickness=thickness)  # Draw in blue
                cv2.polylines(mask, [shape], isClosed=False, color=(0, 0, 0), thickness=thickness)
            elif mode == "add":
                cv2.polylines(image, [shape], isClosed=False, color=(255, 255, 255), thickness=thickness)  # Draw in white
        temp_image = image.copy()
        cv2.imshow("Image", temp_image)

    def draw(event, x, y, flags, param):
        nonlocal points, temp_image, mask, changes_made
        
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            changes_made = True  # Change made, set flag to True
        
        if event == cv2.EVENT_RBUTTONDOWN:
            if len(points) > 1:
                drawn_shapes.append((np.array(points), thickness))  # Store the shape and thickness
                if mode == "delete":
                    cv2.polylines(image, [np.array(points)], isClosed=False, color=(255, 0, 0), thickness=thickness)  # Blue when finalized
                    cv2.polylines(mask, [np.array(points)], isClosed=False, color=(0, 0, 0), thickness=thickness)
                elif mode == "add":
                    cv2.polylines(image, [np.array(points)], isClosed=False, color=(255, 255, 255), thickness=thickness)  # White when finalized
            points = []
            temp_image = image.copy()
            cv2.imshow("Image", image)

        temp_image = image.copy()
        if len(points) > 1:
            # Draw the curve/line in the drawing color
            if mode == "delete":
                cv2.polylines(temp_image, [np.array(points)], isClosed=False, color=(0, 255, 0), thickness=thickness)  # Green while drawing
            elif mode == "add":
                cv2.polylines(temp_image, [np.array(points)], isClosed=False, color=(255, 255, 0), thickness=thickness)  # Cyan while drawing
        cv2.imshow("Image", temp_image)

    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", draw)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if mode == "delete" and changes_made:
                mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
                original_image[mask == 0] = (0, 0, 0)
                cv2.imwrite(output_file, original_image)
            elif mode == "add":
                image = cv2.resize(image, (original_width, original_height))  # Resize back to original size if scaled
                cv2.imwrite(output_file, image)
            print(f"Image saved as '{output_file}'.")
            break
        elif key == ord('u'):  # Undo the last action
            if drawn_shapes:
                drawn_shapes.pop()  # Remove the last drawn shape
                redraw_image()  # Redraw the image
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    return changes_made

def process_files(csv_folder_path, output_dir, scale, overlap_threshold):
    csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.csv')]
    
    print("Left-click to add points to the current curve.")
    print("Right-click to finish the current curve (will change color) and start a new one.")
    print("Press 's' to save the image and apply the mask.")
    print("Press 'u' to undo the last action.")
    print("Press 'q' to exit without saving.")
    
    for csv_file in csv_files:
        csv_file_path = os.path.join(csv_folder_path, csv_file)
        df = pd.read_csv(csv_file_path)
        
        width, height = get_camera_size_from_filename(csv_file)
        
        # Set the minimum area threshold based on camera dimensions
        if width == 1920 and height == 1080:
            min_area_threshold = 3000  # Example value for large cameras
        elif width == 320 and height == 240:
            min_area_threshold = 100  # Example value for small cameras
        else:
            min_area_threshold = 1000  # Default value for other sizes
        
        scaled_boxes = scale_bounding_boxes(df, scale)
        mask = create_overlap_mask(scaled_boxes, width, height, overlap_threshold)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the original mask
        output_file = os.path.join(output_dir, f'{os.path.splitext(csv_file)[0]}.jpg')
        cv2.imwrite(output_file, mask * 255)
        print(f"Saved mask to {output_file}")

        # Draw curves for deletion and save directly as the original file
        loaded_image = cv2.imread(output_file, cv2.IMREAD_COLOR)
        draw_and_save(loaded_image, output_file, width, height, mode="delete")

        # Reload the image after delete operation and save the add version as the original file
        modified_image = cv2.imread(output_file)
        draw_and_save(modified_image, output_file, width, height, mode="add")

        print(f"Final image saved as '{output_file}'.")

        # Apply the post-processing to the final image
        mask = cv2.imread(output_file, cv2.IMREAD_GRAYSCALE)
        
        # Threshold the image to binary if needed
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        filtered_mask_file = os.path.join(output_dir, f'{os.path.splitext(csv_file)[0]}.jpg')
        cv2.imwrite(filtered_mask_file, filtered_mask)
        print(f"Filtered mask saved as '{filtered_mask_file}'.")

# Example usage
csv_folder_path = 'centroidslimit/'  # Folder containing your CSV files
output_dir = 'specified_output_dir/'  # Specify your output directory here
scale = 0.35  # Single scale value
overlap_threshold = 11  # Single threshold value
process_files(csv_folder_path, output_dir, scale, overlap_threshold)
