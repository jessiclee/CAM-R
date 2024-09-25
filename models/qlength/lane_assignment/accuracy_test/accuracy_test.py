import json
import os
import csv

# Function to compare two sets of values and calculate true positives and total
def compare_json(ground_truth, predictions):
    tp = 0
    total = 0

    # Iterate over the ground truth keys
    for key, ground_truth_values in ground_truth.items():
        prediction_values = predictions.get(key, [])
        
        # Count the true positives (correct predictions) by comparing lists
        true_positives = [val for val in prediction_values if val in ground_truth_values]
        tp += len(true_positives)

        # The total number of true positives and ground truth entries
        total += len(ground_truth_values)

    # Return true positives and total count
    return tp, total

# Helper function to get the base name (using split by '_')
def get_base_name(file_name):
    try:
        return file_name.split('_')[0]
    except IndexError:
        return file_name

# Function to process multiple ground truth and prediction comparisons
def process_multiple_comparisons(ground_truth_files, prediction_files, output_csv_file):
    # Create a dictionary of ground truth files based on the base name
    ground_truth_dict = {get_base_name(os.path.splitext(gt)[0]): gt for gt in ground_truth_files}

    # Open the CSV file for writing
    with open(output_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["road", "accuracy", "tp", "total"])

        # Loop through the prediction files and match with ground truth files
        for prediction_file in prediction_files:
            pred_base_name = get_base_name(os.path.splitext(prediction_file)[0])

            # Try to match the prediction with the corresponding ground truth file
            ground_truth_file = ground_truth_dict.get(pred_base_name)

            if ground_truth_file:
                try:
                    # Load the ground truth and prediction files
                    with open(ground_truth_file, 'r') as gt_file:
                        ground_truth = json.load(gt_file)

                    with open(prediction_file, 'r') as pred_file:
                        predictions = json.load(pred_file)

                    # Compare the ground truth and prediction
                    tp, total = compare_json(ground_truth, predictions)

                    # Calculate accuracy
                    accuracy = tp / total if total > 0 else 0

                    # Extract the filename (without extension) to use as the 'road' value
                    road = os.path.splitext(os.path.basename(prediction_file))[0]

                    # Write the row data for each comparison
                    writer.writerow([road, accuracy, tp, total])

                except Exception as e:
                    print(f"Error processing {prediction_file} and {ground_truth_file}: {e}")

# Usage
ground_truth_folder = 'ground_truth'
prediction_folder = 'predictions'

# List of ground truth and prediction files
ground_truth_files = [os.path.join(ground_truth_folder, f) for f in os.listdir(ground_truth_folder) if f.endswith('.json')]
prediction_files = [os.path.join(prediction_folder, f) for f in os.listdir(prediction_folder) if f.endswith('.json')]

output_csv_file = 'lane_assignment_accuracy_results.csv'

process_multiple_comparisons(ground_truth_files, prediction_files, output_csv_file)
