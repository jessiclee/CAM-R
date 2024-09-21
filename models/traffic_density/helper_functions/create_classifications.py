import os
import shutil
import pandas as pd

# Load the CSV file
csv_file = '../image_classificationA.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file)

# Define the base directory where images are stored and where to move them
base_dir = '../photos'  # Replace with the base directory of your images
output_dir = '../end'  # Replace with the directory where you want to move the images

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate through the DataFrame
for index, row in df.iterrows():
    image_name = row['image']  # The column name in CSV containing image names
    classification = row['classification']  # The column name in CSV containing classifications
    
    # Define the path to the image
    src_image_path = os.path.join(base_dir, image_name)
    
    # Create the classification directory if it doesn't exist
    class_dir = os.path.join(output_dir, classification)
    os.makedirs(class_dir, exist_ok=True)
    
    # Define the destination path
    dest_image_path = os.path.join(class_dir, image_name)
    
    # Move the image to the classification directory
    if os.path.exists(src_image_path):
        shutil.move(src_image_path, dest_image_path)
    else:
        print(f"Image {image_name} not found in {base_dir}.")

print("Images have been moved to their respective classification folders.")
