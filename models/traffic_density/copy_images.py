import os
import shutil
import fnmatch

# Define the source directory and destination directory
source_dir = 'D:/CAM-R/images/images/KALLANG PAYA LEBAR EXPRESSWAY/5798'  # Replace with the path to your source directory
dest_dir = 'C:/Users/Jess/OneDrive - Singapore Management University/FYP/density_images/jess_round1'  # Replace with the path to your destination directory

# Ensure the destination directory exists
os.makedirs(dest_dir, exist_ok=True)

# Define the filename patterns you want to match
patterns = [
    '*_01-06-2024_13-*',
    '*_01-06-2024_14-*',
    '*_01-06-2024_15-00-*',
    '*_01-06-2024_19-*',
    '*_01-06-2024_20-00-*',
    '*_01-06-2024_23-*'
]

# Walk through the directory and its subdirectories
for root, dirs, files in os.walk(source_dir):
    for file in files:
        # Check if the file matches any of the patterns
        if any(fnmatch.fnmatch(file, pattern) for pattern in patterns):
            # Define the full source and destination paths
            src_file_path = os.path.join(root, file)
            dest_file_path = os.path.join(dest_dir, file)
            
            # Copy the file to the destination directory
            shutil.copy2(src_file_path, dest_file_path)
            # print(f"Copied: {file} to {dest_dir}")

print("File copying process completed.")
