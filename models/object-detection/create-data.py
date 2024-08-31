import os
import shutil
import random
from tqdm import tqdm
from imutils import paths
import numpy as np
import argparse
import cv2

def dhash(image, hashSize=8):
	# convert the image to grayscale and resize the grayscale image,
	# adding a single column (width) so we can compute the horizontal
	# gradient
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	resized = cv2.resize(gray, (hashSize + 1, hashSize))
	# compute the (relative) horizontal gradient between adjacent
	# column pixels
	diff = resized[:, 1:] > resized[:, :-1]
	# convert the difference image to a hash and return it
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def remove_dupes(path):
    imagePaths = list(paths.list_images(path))
    hashes = {}
    for imagePath in imagePaths:
        print(imagePath)
        # load the input image and compute the hash
        image = cv2.imread(imagePath)
        h = dhash(image)
        # grab all image paths with that hash, add the current image
        # path to it, and store the list back in the hashes dictionary
        p = hashes.get(h, [])
        p.append(imagePath)
        hashes[h] = p
    
    print("hashes hashed")
    for (h, hashedPaths) in hashes.items():
        # check to see if there is more than one image with the same hash
        if len(hashedPaths) > 1:
            print("dupes detected! the following are duplicates:")
            print(hashedPaths)
            for p in hashedPaths[1:]:
                os.remove(p)
                # Also remove the associated .txt file if it exists
                txt_file = os.path.splitext(p)[0] + ".txt"
                if os.path.exists(txt_file):
                    os.remove(txt_file)
                    print(f"Associated text file {txt_file} removed.")
            print("duplicates removed")
    print("done")
    return

def copy_to(files, img_path_dict, path_img, path_label):
    for filex in tqdm(files):
      if filex == 'classes':
          continue
      image_pth = img_path_dict.get(filex)
      label_pth = os.path.splitext(image_pth)[0] + '.txt'
      shutil.copy2(image_pth,f"{path_img}/" + filex + '.jpg' )
      shutil.copy2(label_pth,f"{path_label}/" + filex + '.txt')

def train_test_split(path,neg_path=None, split = 0.15):
    print("------ PROCESS STARTED -------")
    train_path_img = "./yolo_data/train/images/"
    train_path_label = "./yolo_data/train/labels/"
    val_path_img = "./yolo_data/valid/images/"
    val_path_label = "./yolo_data/valid/labels/"
    test_path_img = "./yolo_data/test/images/"
    test_path_label = "./yolo_data/test/labels/"

    image_paths = list(paths.list_images(path))
    files = list(set([os.path.splitext(os.path.basename(name))[0] for name in paths.list_images(path)])) ## removing duplicate names i.e. counting only number of images
    image_path_dict = {os.path.splitext(os.path.basename(name))[0]: name for name in image_paths}

    print (f"--- This folder has a total number of {len(files)} images---")
    random.seed(42)
    random.shuffle(files)

    test_size = int(len(files) * split)
    train_size = len(files) - (2*test_size)

    ## creating required directories

    os.makedirs(train_path_img, exist_ok = True)
    os.makedirs(train_path_label, exist_ok = True)
    os.makedirs(val_path_img, exist_ok = True)
    os.makedirs(val_path_label, exist_ok = True)
    os.makedirs(test_path_img, exist_ok = True)
    os.makedirs(test_path_label, exist_ok = True)
    
    ### ----------- copying images to train folder
    copy_to(files[:train_size], image_path_dict, train_path_img, train_path_label)

    print(f"------ Training data created with 80% split {len(files[:train_size])} images -------")

    ### copytin images to validation folder
    copy_to(files[train_size:(train_size+test_size)], image_path_dict, val_path_img, val_path_label)

    print(f"------ Validation data created with a total of {len(files[train_size:(train_size+test_size)])} images ----------")
    copy_to(files[-test_size:], image_path_dict, test_path_img, test_path_label)

    print(f"------ Testing data created with a total of {len(files[-test_size:])} images ----------")
    print("------ TASK COMPLETED -------")

## spliting the data into train-test and creating train.txt and test.txt file

### Change path directory to local data
# remove_dupes('C:/Users/User/CAM-R/models/object-detection/training')
train_test_split('C:/Users/User/CAM-R/models/object-detection/training/') 