import os
import csv

# Define the path to the sorted photos directory
sorted_photos_dir = "sorted_photos8702/"  # Replace this with your actual path

# Define the CSV output file
output_csv = "sorted_images8702.csv"

# Open the CSV file for writing
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header row
    writer.writerow(['Class', 'Image Name']) 
    
    # Loop through each class folder (high, medium, low)
    for class_name in os.listdir(sorted_photos_dir):
        class_folder = os.path.join(sorted_photos_dir, class_name)
        
        # Ensure it's a directory
        if os.path.isdir(class_folder):
            # Loop through each image in the class folder
            for image_name in os.listdir(class_folder):
                # Write the class name and image name to the CSV
                writer.writerow([class_name, image_name])

print(f"CSV file '{output_csv}' has been created successfully.")
