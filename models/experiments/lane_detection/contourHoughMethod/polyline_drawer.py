"""
Welcome to Polyline Drawer!

SUMMARY: This Polyline Drawer is built to manually add and delete lanes during QC

DEPENDENCIES: numpy, opencv, os, pandas, tkinter

INSTRUCTIONS:
to draw new polyline: click to add points then click n
to delete polyline: simply click on the line 

keys:
b - delete mode
e - exit delete mode
n - draw new polyline/finish the current polyline
d - done
f - finish and close window
"""

##############################################################################
###########################  IMPORTS  ########################################
##############################################################################
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import messagebox

##############################################################################
########################### VARIABLES ########################################
##############################################################################

done = False
points = []
lines = []  # List to store all polygons (detected and newly drawn)
current = (0, 0)
line_selected = -1  # Store the index of the polygon to delete
delete_mode = False     # Set to True when 'b' key is pressed
drawing_mode = True     # True when drawing a new polygon


##############################################################################
########################### FUNCTIONS ########################################
##############################################################################

# Function for saving polygons to file
def save_lines_to_file(lines, file_path):
    with open(file_path, 'w') as file:
        for line in lines:
            # Convert each polygon to a string with floating-point numbers
            line_str = ' '.join(f"{x:.6f},{y:.6f}" for (x, y) in line)
            # Write the polygon to the file
            file.write(line_str + '\n')

# Function to show a message popup
def show_popup(message):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    messagebox.showinfo("Mode", message)
    root.destroy()

# Function to get screen size
def get_screen_size():
    root = tk.Tk()
    root.withdraw() 
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return (width, height)

# Function to check if the user is clicking on the line to delete
def pointLineTest(polyline, point, threshold=4.0):
    px, py = point
    min_distance = float('inf')  # Start with a large value

    # Iterate over each pair of consecutive points in the polyline
    for i in range(len(polyline) - 1):
        (x1, y1) = polyline[i]
        (x2, y2) = polyline[i + 1]

        # Calculate distance from point to the current line segment
        num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
        den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        
        distance = num / den if den != 0 else float('inf')
        
        # Track the minimum distance found
        min_distance = min(min_distance, distance)
    
    # Return True if the closest distance is within the threshold
    return min_distance <= threshold

# Function to load auto polylines from txt file
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

# Function to define what happens for every user input
def on_mouse(event, x, y, buttons, user_param):
    global done, points, current, temp, lines, line_selected, delete_mode, drawing_mode

    if delete_mode and event == cv2.EVENT_LBUTTONDOWN:
        # Check if clicked inside any polygon
        for i, line in enumerate(lines):
            if pointLineTest(line, (x, y)):
                print(f"line {i} selected for deletion.")
                line_selected = i
                break

        if line_selected != -1:
            # Delete the selected polygon
            del lines[line_selected]
            line_selected = -1
            # Clear and update the image
            temp = clone.copy()
            for poly in lines:
                cv2.polylines(temp, [poly], isClosed=False, color=(255, 0, 0), thickness=2)

            # Redraw the current polygon-in-progress
            if len(points) > 1:
                cv2.polylines(temp, [np.array(points)], False, (0, 0, 255), 1)
                cv2.line(temp, (points[-1][0], points[-1][1]), current, (0, 0, 255))
            cv2.imshow("image", temp)
        return

    if drawing_mode:
        if event == cv2.EVENT_MOUSEMOVE:
            current = (x, y)

        elif event == cv2.EVENT_LBUTTONDOWN:
            # Adding points for a new polygon
            print(f"Adding point #{len(points)} at position ({x}, {y})")
            cv2.circle(img, (x, y), 5, (0, 200, 0), -1)
            points.append([x, y])
            temp = img.copy()
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right-click completes the polygon
            if len(points) > 1:
                print(f"Completing line with {len(points)} points.")
                lines.append(np.array(points))  # Save current polygon
                points = []  # Reset points for a new polygon
                temp = img.copy()
            else:
                print("Need at least 2 points to complete a line")


##############################################################################
###########################DRIVER CODE########################################
##############################################################################

# ***** replace with required image path *****
roadNum = input("Enter road ID: ")
main_folder_dir = "C:/Users/Zhiyi/Desktop/FYP/newtraffic/"
path = main_folder_dir + "images/" + roadNum + ".jpg"
img = cv2.imread(path)
clone = img.copy()
temp = img.copy()
final_img = img.copy()

lines = load_lines_from_file(main_folder_dir + "v3result/autolines/" + roadNum + '.txt')

cv2.namedWindow("image", cv2.WINDOW_NORMAL)
screen_width, screen_height = get_screen_size()
cv2.resizeWindow("image", screen_width, screen_height)
cv2.setMouseCallback("image", on_mouse)

while True:
    img = temp.copy()
    # Draw all the saved polygons (detected and newly drawn)
    for line in lines:
        line = line.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(img, [line], isClosed=False, color=(255, 0, 0), thickness=2)

    # Draw the current polygon-in-progress
    if len(points) > 1:
        cv2.polylines(img, [np.array(points)], False, (0, 0, 255), 1)
        cv2.line(img, (points[-1][0], points[-1][1]), current, (0, 0, 255))

    # Update the window
    cv2.imshow("image", img)


    # Check for user input
    key = cv2.waitKey(50) & 0xFF
    if key == ord('d'):  # Press 'd' when done with all polygons
        print("Finalizing lines...")
        break

    elif key == ord('n'):  # Press 'n' to start a new polygon
        if len(points) > 1:
            lines.append(np.array(points))
            points = []
            temp = img.copy()
            print("Starting new line.")
        else:
            print("Need at least 2 points to complete a line.")

    elif key == ord('r'):  # Press 'r' to reset the current in-progress polygon
        show_popup("Try drawing again")
        points = []
        temp = clone.copy()
        print("Reset current line.")

    elif key == ord('b'):  # Press 'b' to enter delete mode
        show_popup("Delete line mode!")
        delete_mode = True
        drawing_mode = False
        print("Delete mode: click on a line to delete it.")

    elif key == ord('e'):  # Press 'e' to exit delete mode
        show_popup("Exit delete mode")
        delete_mode = False
        drawing_mode = True
        print("Exit delete mode.")

# Final drawing of all polygons
for line in lines:
    line = line.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(final_img, [line], isClosed=False, color=(255, 0, 0), thickness=2)

# Show final image
cv2.imshow("Line Drawer", final_img)
cv2.waitKey(0)

# Save the image with polygons
cv2.imwrite(main_folder_dir + "v3result/manual/mask-" + os.path.basename(path), final_img)
save_lines_to_file(lines,  main_folder_dir + "v3result/manual/lines/" + roadNum + '.txt')
cv2.destroyAllWindows()


