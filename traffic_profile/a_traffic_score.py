import csv
import json

import numpy as np

#get density
def get_density_score(camid):
    #if low  return 0.25
    #return 
    return 5

#get count
def get_queue_length(camid, laneid):

    #return queue_length
    return 5

# # Open the CSV file to write data
# with open('traffic_score.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['img', 'lane_grouping', 'density_score', 'total_count', 'count_percentile', 'overallscore'])

#     #load lane grouping
#     lane_groupings = {}
#     with open("lane_grouping.json", 'r') as file:
#         lane_groupings = json.load(file)

#     #compute total score  and density score
#     for img in lane_groupings:
#         density_score = get_density_score(img)
#         for group in img:  
#             total_count = 0
#             for lane in group:
#                 queue_length = get_queue_length(img, lane)
#                 total_count += queue_length
#             writer.writerow([img, group, density_score, total_count, '', ''])
# Open the CSV file to write data
with open('traffic_score.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['img', 'lane_grouping', 'density_score', 'total_count', 'count_percentile', 'overallscore'])

    # Load lane grouping
    lane_groupings = {}
    with open("lane_grouping.json", 'r') as file:
        lane_groupings = json.load(file)

    # Compute total score and density score
    for img, groups in lane_groupings.items():  # img is cam_id, groups is the dictionary of group_nums
        density_score = get_density_score(img)  # Assuming this function returns a density score
        for group_num, lanes in groups.items():  # group_num is "group_num1", "group_num2", etc., lanes is the list of lane IDs
            total_count = 0
            for lane in lanes:  # Iterate over lanes within the group
                queue_length = get_queue_length(img, lane)  # Assuming this function returns a queue length
                total_count += queue_length
            # Write row with the image (cam_id), group_num, density score, and total count
            writer.writerow([img, group_num, density_score, total_count, '', ''])


#compute count_percentile
total_counts = []
rows = []

with open('traffic_score.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        total_counts.append(float(row[3]))  # Collect total_count (column index 3)
        rows.append(row)  # Store each row for later update


#percentiles = [np.percentile(total_counts, (np.sum(total_counts < x) / len(total_counts)) * 100) for x in total_counts]
percentiles = [np.percentile(total_counts, (np.sum(np.array(total_counts) < x) / len(total_counts)) * 100) for x in total_counts]


with open('traffic_score.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['img', 'lane_grouping', 'density_score', 'total_count', 'count_percentile', 'overallscore'])

    
    for i, row in enumerate(rows):
        row[4] = percentiles[i]  
        writer.writerow(row)

#compute overall score
count_percentiles = []
density_scores = []
rows = []

# Read the CSV file and collect scores
with open('traffic_score.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        density_scores.append(float(row[2]))  # Collect density_score (column index 2)
        count_percentiles.append(float(row[4]))  # Collect count_score (column index 4)
        rows.append(row)  # Store each row for later update

# Compute overall_score as the average of density_score and count_score
overall_scores = np.array(density_scores) / 2 + np.array(count_percentiles) / 2

# Write back the results, including the overall_score
with open('traffic_score.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['img', 'lane_grouping', 'density_score', 'total_count', 'count_percentile', 'overallscore'])

    for i, row in enumerate(rows):
        row[5] = overall_scores[i]  # Update overall_score in column index 5
        writer.writerow(row)

#ouput csv for each day each cam each group