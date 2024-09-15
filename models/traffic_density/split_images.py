import splitfolders

# Define the path to your dataset
input_folder = 'end/'

# Define the output folder where the split data will be saved
output_folder = 'split_data/'

# Perform the split
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.7, .15, .15), group_prefix=None)
