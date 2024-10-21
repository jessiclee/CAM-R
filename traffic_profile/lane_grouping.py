import pandas as pd
import json

# Load the CSV file
file_path = 'lane_groups.csv'  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Group the dataframe by 'cam_id' and 'group_num', and aggregate 'lane_id' values as lists
grouped = df.groupby(['cam_id', 'group_num'])['lane_id'].apply(list).reset_index()

# Initialize an empty dictionary to hold the desired structure
result = {}

# Iterate over the grouped data to populate the dictionary
for _, row in grouped.iterrows():
    cam_id = str(row['cam_id'])
    group_num = f"group_num{row['group_num']}"
    
    if cam_id not in result:
        result[cam_id] = {}
    
    # Assign group_num directly as a key with the list of lanes as the value
    result[cam_id][group_num] = row['lane_id']

# Convert the result to JSON format
json_output = json.dumps(result, indent=4)

# Save the JSON to a file
output_file_path = 'lane_grouping.json'  # Specify the path to save the output JSON
with open(output_file_path, 'w') as json_file:
    json_file.write(json_output)

print(f"JSON file saved at: {output_file_path}")
