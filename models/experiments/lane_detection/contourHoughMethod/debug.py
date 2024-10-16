##############################################################################
###########################  IMPORTS  ########################################
##############################################################################
import pandas as pd
import os
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
import tkinter as tk
from tkinter import messagebox
camera_sizes = {
    "small": {
        "width": 320,
        "height": 240,
        "cameras": [1001, 1004, 1005, 1006, 1504, 1501, 1502, 1503, 1505, 1002, 1003]
    },
    "large": {
        "width": 1920,
        "height": 1080,
        "cameras": [4701, 4702, 4704, 4705, 4706, 4707, 4708, 4709, 4710, 4712, 4714, 
                    4716, 4798, 4799, 6716, 2703, 2704, 2705, 2706, 2707, 2708, 1701, 
                    1702, 1703, 1704, 1705, 1706, 1707, 1709, 1711, 1113, 3702, 3705, 
                    3793, 3795, 3796, 3797, 3798, 3704, 5798, 8701, 8702, 8704, 8706, 
                    5794, 5795, 5797, 5799, 6701, 6703, 6704, 6705, 6706, 6708, 6710, 
                    6711, 6712, 6713, 6714, 6715, 7797, 7798, 9701, 9702, 9703, 9704, 
                    9705, 9706, 1111, 1112, 7791, 7793, 7794, 7795, 7796, 4703, 4713, 
                    2701, 2702]
    }
}

# Function to get screen size
def get_screen_size():
    root = tk.Tk()
    root.withdraw() 
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return (width, height)

def load_lines_from_file(file_path):
    lines = []
    # Check if the file exists
    if not os.path.exists(file_path):
        # If the file doesn't exist, simply return an empty list or handle it as needed
        return lines

    with open(file_path, 'r') as file:
        for line in file:
            # Convert each line to a list of tuples, handling floating-point numbers
            points = [tuple(map(float, pt.split(','))) for pt in line.strip().split()]
            # Convert the list of tuples to a numpy array and ensure correct data type
            lines.append(np.array(points, dtype=np.float32))
    return lines

def get_camera_size(camid):
    camera_id = int(camid)
    
    # Find the corresponding size based on camera ID
    if camera_id in camera_sizes['small']['cameras']:
        width = camera_sizes['small']['width']
        height = camera_sizes['small']['height']
    elif camera_id in camera_sizes['large']['cameras']:
        width = camera_sizes['large']['width']
        height = camera_sizes['large']['height']
    else:
        raise ValueError(f"Camera ID {camera_id} from file '{filename}' is not predefined.")
    
    return width, height
# ***** replace with required image path *****
roadNum = input("Enter road ID: ")
main_folder_dir = "C:/Users/Zhiyi/Desktop/FYP/newtraffic/"
path = main_folder_dir + "centroidimages/" + roadNum + ".jpg"
img = cv2.imread(path)
clone = img.copy()
temp = img.copy()
final_img = img.copy()
img_width, img_height = get_camera_size(roadNum)
is_hd = True

if img_width == 240:
    is_hd = False


lines = load_lines_from_file("C:/Users/Zhiyi/Desktop/FYP/newtraffic/v3result/manual2/lines/" + roadNum + ".txt")

img = temp.copy()

    # Draw all the saved polygons (detected and newly drawn)\
lane_id = 1
for line in lines:
    line = np.array(line)
    line = line.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(img, [line], isClosed=False, color=(255, 0, 0), thickness=2)
    coord1 = int((line[1][0][0] + line[0][0][0]) / 2)
    coord2 = int((line[1][0][1] + line[0][0][1]) / 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"{lane_id}", (coord1 , coord2), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    lane_id +=1
    
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
screen_width, screen_height = get_screen_size()
cv2.resizeWindow("image", screen_width, screen_height)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
