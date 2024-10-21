import numpy as np
import json

#get density
def get_density_score(camid):

    return queue_length


#get count
def get_queue_length(camid, laneid):

    return queue_length


# Step 1: Initialize variables
lane_groupings = {}
total_counts = []

# Load lane grouping from JSON file
with open("lane_groupings.json", 'r') as lane_file:
    lane_groupings = json.load(lane_file)

# Step 2: First pass: Calculate total_count and store only necessary results
results = []  # List to store the final results

for img_idx, img in enumerate(lane_groupings):
    for group_idx, group in enumerate(img):
        density_score = get_density_score(img)
        total_count = 0
        for lane in group:
            queue_length = get_queue_length(img, lane)  
            total_count += queue_length

        # Store img, lane_grouping, and total_count for further calculation
        results.append({
            'img': img,
            'lane_grouping': group_idx,
            'density_score': density_score,
            'total_count': total_count
        })

        total_counts.append(total_count)  # Store total_count for percentile calculation

# Step 3: Calculate percentiles for total_count (count_score is now a percentile)
percentiles = [np.percentile(total_counts, (np.sum(total_counts < x) / len(total_counts)) * 100) for x in total_counts]

# Step 4: Compute overall score (average of total_count and count_score)
overall_scores = [(percentiles[i] + total_counts[i]) / 2 for i in range(len(total_counts))]

# Step 5: Store only img, lane_grouping, and overallscore in the final result
final_results = []

for i, result in enumerate(results):
    final_results.append({
        'img': result['img'],
        'lane_grouping': result['lane_grouping'],
        'overallscore': overall_scores[i]
    })

# Now you have the final results in memory with only img, lane_grouping, and overallscore
for result in final_results:
    print(result)
