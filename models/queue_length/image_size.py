import cv2

def get_image_dimensions(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Get dimensions
    height, width, _ = image.shape
    return width, height

# Example usage
image_path = 'path/to/your/image.jpg'  # Replace with your image path
width, height = get_image_dimensions('C:\\Users\\jesle\\Desktop\\fyp\\actual_data\\queue_length_test\\9702_01-06-2024_13-05-02.jpg')
print(f"Width: {width}, Height: {height}")
