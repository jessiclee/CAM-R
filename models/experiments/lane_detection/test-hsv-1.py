# Based on this paper: https://informatika.stei.itb.ac.id/~rinaldi.munir/Penelitian/Makalah-TSSA-2021.pdf
import cv2
import numpy as np
import os
import glob

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Apply Gaussian blur to the image
    # Determine the kernel size based on the heuristic (3 * sigma, rounded to nearest odd integer)
    sigma = 1.0  # Example sigma value; you might need to adjust this
    kernel_size = int(3 * sigma)
    if kernel_size % 2 == 0:
        kernel_size += 1

    blurred_image = cv2.GaussianBlur(hsv_image, (kernel_size, kernel_size), sigma)

    return blurred_image

def segment_road(image):
    # Define HSV range for road segmentation
    # These values might need adjustment based on the specific road and lighting conditions
    lower_hsv = np.array([0, 0, 50])  # Lower bound of HSV for road
    upper_hsv = np.array([180, 50, 255])  # Upper bound of HSV for road

    # Create a binary mask based on the HSV range
    mask = cv2.inRange(image, lower_hsv, upper_hsv)

    return mask

def post_process_mask(mask):
    # Apply erosion and dilation
    kernel = np.ones((5, 5), np.uint8)

    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)

    return dilated_mask

def extract_dominant_segment(mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour by area
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Create a new mask with only the largest contour
        dominant_mask = np.zeros_like(mask)
        cv2.drawContours(dominant_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        return dominant_mask
    else:
        return mask  # Return the original mask if no contours are found

def apply_mask(image_path, mask):
    # Load the original image
    original_image = cv2.imread(image_path)

    # Convert mask to 3 channels
    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Apply mask to the original image
    masked_image = cv2.bitwise_and(original_image, mask_3channel)

    return masked_image

def detect_lanes(masked_image):
    # Convert the masked image to grayscale
    gray_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    # Use Canny edge detection
    edges = cv2.Canny(gray_image, 50, 150)

    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)

    lane_count = 0
    if lines is not None:
        # Draw lines on the image (optional, for visualization)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(masked_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            lane_count += 1

    return masked_image, lane_count

def process_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    road_mask = segment_road(preprocessed_image)
    refined_mask = post_process_mask(road_mask)
    dominant_mask = extract_dominant_segment(refined_mask)

    return dominant_mask

if __name__ == "__main__":
    root_dir = 'test_images'
    images = []
    images.extend(glob.glob(os.path.join(root_dir, '**', '*.jpg'), recursive=True))
    
    for i in images:
        image_name = os.path.basename(i)
        image_path = i
        dominant_mask = process_image(image_path)

        # Save masks
        # cv2.imwrite('binary_masks/' + image_name, dominant_mask)

        # Apply the mask to the original image and save image
        masked_image = apply_mask(image_path, dominant_mask)
        cv2.imwrite('masks/' + image_name, masked_image)
        
        # Detect lanes
        # masked_image_with_lanes, lane_count = detect_lanes(masked_image)
        # cv2.imwrite('lane_roi/' + image_name, masked_image_with_lanes)
        # print(f'Number of lanes detected: {lane_count}')
        
        

