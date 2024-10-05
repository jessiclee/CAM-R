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
        
        # Sort the lists to handle unordered comparison
        ground_truth_values = sorted(ground_truth_values)
        prediction_values = sorted(prediction_values)

        # Count the true positives (correct predictions) by comparing lists
        true_positives = [val for val in prediction_values if val in ground_truth_values]
        tp += len(true_positives)

        # The total number of ground truth entries
        total += len(ground_truth_values)

    # Return true positives and total count
    return tp, total

# Helper function to get the base name (without file extension)
def get_base_name(file_name):
    # Return the filename without extension
    return os.path.splitext(os.path.basename(file_name))[0]

# Function to process multiple ground truth and prediction comparisons
def process_multiple_comparisons(ground_truth_files, prediction_files, output_csv_file):
    # Create a dictionary of ground truth files based on the base name
    ground_truth_dict = {get_base_name(gt): gt for gt in ground_truth_files}

    with open(output_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["road", "accuracy", "tp", "total"])

        for prediction_file in prediction_files:
            pred_base_name = get_base_name(prediction_file)
            ground_truth_file = ground_truth_dict.get(pred_base_name)

            if ground_truth_file:
                try:
                    # Load ground truth file
                    with open(ground_truth_file, 'r') as gt_file:
                        ground_truth = json.load(gt_file)

                    # Load prediction file
                    with open(prediction_file, 'r') as pred_file:
                        predictions = json.load(pred_file)

                    # Compare the ground truth and prediction
                    tp, total = compare_json(ground_truth, predictions)

                    # Calculate accuracy and round to 4 decimal places
                    accuracy = round(tp / total, 4) if total > 0 else 0.0

                    # Write the row data for each comparison without the "overlap\" prefix
                    road = get_base_name(prediction_file)
                    writer.writerow([road, accuracy, tp, total])

                except json.JSONDecodeError as e:
                    print(f"JSON decode error in {ground_truth_file} or {prediction_file}: {e}")
                except Exception as e:
                    print(f"Error processing {prediction_file} and {ground_truth_file}: {e}")
            else:
                print(f"No matching ground truth for {prediction_file}")

# Usage
ground_truth_folder = 'ground_truth'
prediction_folder = 'contour'

# List of ground truth and prediction files
ground_truth_files = [os.path.join(ground_truth_folder, f) for f in os.listdir(ground_truth_folder) if f.endswith('.json')]
prediction_files = [os.path.join(prediction_folder, f) for f in os.listdir(prediction_folder) if f.endswith('.json')]

# Debug: Check that folders contain the expected files
# print("Ground truth folder contents:", ground_truth_files)
# print("Prediction folder contents:", prediction_files)

output_csv_file = 'lane_assignment_accuracy_results_contour.csv'

# Run the comparison process
process_multiple_comparisons(ground_truth_files, prediction_files, output_csv_file)
