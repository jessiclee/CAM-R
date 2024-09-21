"""
UPDATE: THIS DOES NOT WORK
Through the polygons, get the median line in the contour and then extrapolate the line.
Given a centroid, get the euclidean distance then assign it to the lane that it is the closest to.
"""
# IMPORTS
import cv2
import numpy as np
import os
from skimage import morphology
from skimage.util import invert
from skimage import data
import tkinter as tk
from skan import csr
from scipy.special import binom
from skimage import img_as_ubyte

# GLOBAL VARIABLES
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

# FUNCTIONS 
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

def load_polygons_from_file(file_path):
    polygons = []
    # Check if the file exists
    if not os.path.exists(file_path):
        print("file path does not exist")
        # If the file doesn't exist, simply return an empty list or handle it as needed
        return polygons

    with open(file_path, 'r') as file:
        for line in file:
            # Convert each line to a list of tuples, handling floating-point numbers
            points = [tuple(map(float, pt.split(','))) for pt in line.strip().split()]
            # Convert the list of tuples to a numpy array and ensure correct data type
            polygons.append(np.array(points, dtype=np.float32))
    return polygons

def save_lines(image, file_path):
    pass

def skeletonize_image(image):
    skeleton = morphology.skeletonize(image, method='lee')
    skeleton_uint8 = np.array(skeleton * 255, dtype=np.uint8)
    return skeleton_uint8



def get_vehicles_from_csv(image):
    pass

def line_assignment_by_perpendicular_distance(centroid, lineID):
    pass

def save_line_assignment(lineID, bboxes):
    pass

# Function to get screen size
def get_screen_size():
    root = tk.Tk()
    root.withdraw() 
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return (width, height)

def Bernstein(n, k):
    """Bernstein polynomial.
    """
    coeff = binom(n, k)

    def _bpoly(x):
        return coeff * x ** k * (1 - x) ** (n - k)

    return _bpoly

def Bezier(points, num=1000):
    """Build Bézier curve from points with improvements for smoothness and size retention."""
    N = len(points)  # Number of control points
    t = np.linspace(0, 1, num=num)  # Parameter t from 0 to 1 for smooth interpolation
    curve = np.zeros((num, 2))  # Initialize the curve array with shape (num, 2)

    # Compute Bézier curve by summing Bernstein polynomials
    for ii in range(N):
        curve += np.outer(Bernstein(N - 1, ii)(t), points[ii])
    
    # Ensure the curve fits within the original polygon bounds (retain size)
    min_points = np.min(points, axis=0)  # Find min x and y from original points
    max_points = np.max(points, axis=0)  # Find max x and y from original points
    min_curve = np.min(curve, axis=0)
    max_curve = np.max(curve, axis=0)
    
    # Scale the curve back to fit the original polygon size
    scale = (max_points - min_points) / (max_curve - min_curve)
    curve = (curve - min_curve) * scale + min_points

    return curve


# input road
roadNum = input("Enter road ID: ")

# directory paths
main_folder_dir = "C:/Users/Zhiyi/Desktop/FYP/newtraffic/"
image_path = main_folder_dir + "images/" + roadNum + ".jpg"
lane_polygons_path = main_folder_dir + "v2result/manual/polygons/" + roadNum + ".txt"

# get the saved polygons from the image and draw the polygons on a black image
polygons = load_polygons_from_file(lane_polygons_path)
road_image = cv2.imread(image_path)
width, height = get_camera_size(roadNum)
lane_lines_image = np.zeros((height, width), dtype=np.uint8)
skeleton_lanes = []
skeletonized = None
res = None
# get the median line in each polygon.
## CURRENT V1
for poly in polygons:
    lane_polygon_image = np.zeros((height, width), dtype=np.uint8)
    # poly = Bezier(poly)
    poly = poly.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(lane_polygon_image, [poly], color=255)
    # use scikit-image skeletonize to get the lane lines
    skeletonized = skeletonize_image(lane_polygon_image)
    skeleton_coords = np.column_stack(np.where(skeletonized > 0))
    skeleton_lanes.append(skeleton_coords)

# Plot the skeleton coordinates on the new image
for skeleton_coords in skeleton_lanes:
    for coord in skeleton_coords:
        cv2.circle(lane_lines_image, (coord[1], coord[0]), radius=1, color=255, thickness=-1)

cv2.namedWindow("lines", cv2.WINDOW_NORMAL)
screen_width, screen_height = get_screen_size()
cv2.resizeWindow("lines", screen_width, screen_height)
cv2.imshow("lines", lane_polygon_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# load the centroids and lines

# calculate the perpendicular distance from a vehicle to each line. Get the closest line and assign the vehicle to that road. 