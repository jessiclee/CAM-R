import csv
import json

import numpy as np

#get density
#get count
def get_queue_length(camid, laneid):

    return queue_length

# Open the CSV file to write data
with open('traffic_score.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['img', 'lane_grouping', 'density_score', 'total_count', 'count_score', 'overallscore'])

#load lane grouping
lane_groupings = {}
with open("/lane_groupings.json", 'r') as file:
    lane_groupings = json.load(file)

#compute total score
for img in lane_groupings:
    for group in img:
        total_count = 0
        for lane in group:
            queue_length = get_queue_length(img, lane)
            total_count += queue_length
    writer.writerow([img, group, '', total_count, '', ''])

#compute count score
total_counts = []
rows = []

with open('traffic_score.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        total_counts.append(float(row[3]))  # Collect total_count (column index 3)
        rows.append(row)  # Store each row for later update


percentiles = [np.percentile(total_counts, (np.sum(total_counts < x) / len(total_counts)) * 100) for x in total_counts]


with open('traffic_score.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['img', 'lane_grouping', 'density_score', 'total_count', 'count_score', 'overallscore'])

    
    for i, row in enumerate(rows):
        row[4] = percentiles[i]  
        writer.writerow(row)

#compute overall score
count_scores = []
density_scores = []
rows = []

# Read the CSV file and collect scores
with open('traffic_score.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        density_scores.append(float(row[2]))  # Collect density_score (column index 2)
        count_scores.append(float(row[4]))  # Collect count_score (column index 4)
        rows.append(row)  # Store each row for later update

# Compute overall_score as the average of density_score and count_score
overall_scores = np.array(density_scores) / 2 + np.array(count_scores) / 2

# Write back the results, including the overall_score
with open('traffic_score.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['img', 'lane_grouping', 'density_score', 'total_count', 'count_score', 'overallscore'])

    for i, row in enumerate(rows):
        row[5] = overall_scores[i]  # Update overall_score in column index 5
        writer.writerow(row)

#ouput csv for each day each cam each group