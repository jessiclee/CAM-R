import os
import shutil
from tqdm import tqdm
from imutils import paths

official_classses = {
    "bus":0,
    "truck":1,
    "motorcycle":2,
    "car":3
}
official_class2 = {
    "car_back":3,
    "car_left":3,
    "car_front":3,
    "car_right":3,
    "car":3,
    "bus_back":0,
    "bus_left":0,
    "bus_front":0,
    "bus_right":0,
    "bus":0,
    "truck_back":1,
    "truck_left":1,
    "truck_front":1,
    "truck_right":1,
    "truck":1,
    "motorcycle_back":2,
    "motorcycle_front":2,
    "motorcycle_left":2,
    "motorcycle_right":2,
    "motorcycle":2,
}

# Define paths
labels = "./train/labels"
images = "./train/images"
image_paths = "./train2/obj_train_data/"  # Folder containing images

# Get list of image paths
image_paths_list = list(paths.list_images(image_paths))

# Create list of unique base names of the images
files3 = list(set([os.path.splitext(os.path.basename(name))[0] for name in image_paths_list]))

# Create a dictionary mapping base names to full image paths
image_path_dict = {os.path.splitext(os.path.basename(name))[0]: name for name in image_paths_list}

# Define the copy_to function
def copy_to(files, img_path_dict, path_img, path_label):
    for filex in tqdm(files):
        if filex == 'classes':
            continue
        image_pth = img_path_dict.get(filex)
        label_pth = os.path.splitext(image_pth)[0] + '.txt'
        shutil.copy2(image_pth, f"{path_img}/" + filex + '.jpg')
        shutil.copy2(label_pth, f"{path_label}/" + filex + '.txt')

# Create directories if they don't exist
os.makedirs(labels, exist_ok=True)
os.makedirs(images, exist_ok=True)

# Call the function with the correct parameters
copy_to(files3, image_path_dict, images, labels)

print("Finished")


# if os.path.isfile(file_path):
#     with open(file_path, 'r') as file:
#         # content = file.read()
#         # print(content)  # Print the file content
#         for index, line in enumerate(file):
#         # Strip whitespace from each line and add to dictionary
#             item = line.strip()
#             data_dict[index] = item  # Using 1-based indexing
        
# for file in os.listdir(image_paths):
#     file_path2 = os.path.join(image_paths, file)
#     if file.endswith('.txt') and file != "train.txt" and os.path.getsize(file_path2) > 0:
#         with open(file_path2, 'r') as f:
#             lines = f.readlines()
#             modified_lines = []
#             for line in lines:
#                 parts = line.strip().split()  # Split line into part
#                 if parts:
#                     print(parts)
#                     obj_class = data_dict.get(int(parts[0])) 
#                     print("original class and number", parts[0], obj_class)
#                     real_num = official_class2.get(obj_class)
#                     print("actual num", real_num)
#                     parts[0] = str(real_num)
#                     modified_lines.append(' '.join(parts))
#         with open(file_path, 'w') as f:
#             print("changed lines: ", modified_lines)
#             f.write('\n'.join(modified_lines))
