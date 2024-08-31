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

def train_test_split(path,neg_path=None, split = 0.15):
    print("------ PROCESS STARTED -------")
    train_path_img = "./yolo_data/train/images/"
    train_path_label = "./yolo_data/train/labels/"
    val_path_img = "./yolo_data/valid/images/"
    val_path_label = "./yolo_data/valid/labels/"
    test_path = "./yolo_data/test/images/"
    test_path_label = "./yolo_data/test/labels/"


    files = list(set([name[:-4] for name in os.listdir(path)])) ## removing duplicate names i.e. counting only number of images
    

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
    for filex in tqdm(files[:train_size]):
      if filex == 'classes':
          continue
      shutil.copy2(path + filex + '.jpg',f"{train_path_img}/" + filex + '.jpg' )
      shutil.copy2(path + filex + '.txt', f"{train_path_label}/" + filex + '.txt')
        
    

    print(f"------ Training data created with 80% split {len(files[:train_size])} images -------")

    ### copytin images to validation folder
    for filex in tqdm(files[train_size:(train_size+test_size)]):
      if filex == 'classes':
          continue
      # print("running")
      shutil.copy2(path + filex + '.jpg', f"{val_path_img}/" + filex + '.jpg' )
      shutil.copy2(path + filex + '.txt', f"{val_path_label}/" + filex + '.txt')

    print(f"------ Validation data created with a total of {len(files[train_size:(train_size+test_size)])} images ----------")

    for filex in tqdm(files[-test_size:]):
      if filex == 'classes':
          continue
      # print("running")
      shutil.copy2(path + filex + '.jpg', f"{val_path_img}/" + filex + '.jpg' )
      shutil.copy2(path + filex + '.txt', f"{val_path_label}/" + filex + '.txt')

    print(f"------ Testing data created with a total of {len(files[-test_size:])} images ----------")
    print("------ TASK COMPLETED -------")

## spliting the data into train-test and creating train.txt and test.txt file

### Change path directory to local data
train_test_split('C:/Users/kwekz/fyp/training-test/data/') ### without negative images