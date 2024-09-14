import os
import pandas as pd

# Define the directory containing the images
image_folder = 'photos'  # Replace with your image folder path

# List all files in the directory
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# Create a DataFrame with image names and an empty classification column
df = pd.DataFrame(image_files, columns=['image'])
df['classification'] = ''  # Add an empty 'classification' column

# Save the DataFrame to a CSV file
csv_output = 'image_classification.csv'  # Replace with your desired output CSV file path
df.to_csv(csv_output, index=False)

print(f"CSV file created at {csv_output} with {len(image_files)} images.")
