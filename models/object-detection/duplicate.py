import os
from PIL import Image
import imagehash

# Directory containing images
image_dir = 'C:\\Users\\jesle\\Desktop\\fyp\\actual_data\\model_evaluation_data\\train\\images'

# Function to find duplicates
def find_duplicate_images(image_dir):
    # Dictionary to store hashes and file paths
    hash_dict = {}

    # List all files in the directory
    for filename in os.listdir(image_dir):
        # Only process image files
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(image_dir, filename)

            # Open the image and compute its hash
            with Image.open(file_path) as img:
                img_hash = imagehash.phash(img)

            # Check if this hash exists in the dictionary
            if img_hash in hash_dict:
                print(f"Duplicate found: {file_path} and {hash_dict[img_hash]}")
            else:
                hash_dict[img_hash] = file_path

# Call the function to detect duplicates
find_duplicate_images(image_dir)
