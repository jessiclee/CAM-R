import os
import shutil
import fnmatch

def copy_files(source_dir, dest_dir, patterns):
    """
    Copy files from source directory to destination directory if they match the specified patterns.
    """
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

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

def delete_files(source_dir, patterns):
    """
    Delete files in the source directory that match the specified patterns.
    """
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # Check if the file matches any of the patterns
            if any(fnmatch.fnmatch(file, pattern) for pattern in patterns):
                # Define the full path to the file
                file_path = os.path.join(root, file)
                
                try:
                    # Delete the file
                    os.remove(file_path)
                    # print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Could not delete {file_path}: {e}")

    print("File deletion process completed.")

# Example usage:
source_dir = 'D:/CAM-R/images/images'  # Replace with the path to your source directory
dest_dir = 'C:/Users/Jess/OneDrive - Singapore Management University/FYP/density_images/abiya_round2'  # Replace with the path to your destination directory
patterns = ['*_01-07-2024_00-*', '*_01-07-2024_23-*', '*_01-07-2024_06-*',  
            ]  # Replace with the patterns you want to match
# patterns = ['*_01-07-2024_*']  # Replace with the patterns you want to match

# Call the functions as needed
# copy_files(source_dir, dest_dir, patterns)  # To copy files
delete_files(dest_dir, patterns)  # To delete files
