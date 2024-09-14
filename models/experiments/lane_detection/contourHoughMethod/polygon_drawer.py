import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import messagebox

# Function for saving polygons to file
def save_polygons_to_file(polygons, file_path):
    with open(file_path, 'w') as file:
        for poly in polygons:
            # Convert each polygon to a string with floating-point numbers
            poly_str = ' '.join(f"{x:.6f},{y:.6f}" for (x, y) in poly.squeeze())
            # Write the polygon to the file
            file.write(poly_str + '\n')

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

# ***** replace with required image path *****
roadNum = input("Enter road ID: ")
main_folder_dir = "C:/Users/Zhiyi/Desktop/FYP/newtraffic/"
# path = main_folder_dir + "result/" + roadNum + ".jpg"
path = main_folder_dir + "images/" + roadNum + ".jpg"

img = cv2.imread(path)
clone = img.copy()
temp = img.copy()
final_img = img.copy()

# ***** global variable declaration *****
done = False
points = []
polygons = []  # List to store all polygons (detected and newly drawn)
current = (0, 0)
polygon_selected = -1  # Store the index of the polygon to delete
delete_mode = False     # Set to True when 'b' key is pressed
drawing_mode = True     # True when drawing a new polygon

def load_polygons_from_file(file_path):
    polygons = []
    # Check if the file exists
    if not os.path.exists(file_path):
        # If the file doesn't exist, simply return an empty list or handle it as needed
        return polygons

    with open(file_path, 'r') as file:
        for line in file:
            # Convert each line to a list of tuples, handling floating-point numbers
            points = [tuple(map(float, pt.split(','))) for pt in line.strip().split()]
            # Convert the list of tuples to a numpy array and ensure correct data type
            polygons.append(np.array(points, dtype=np.float32))
    return polygons

# Load from file
polygons = load_polygons_from_file(main_folder_dir + "v2result/autopolygons/" + roadNum + '.txt');

def on_mouse(event, x, y, buttons, user_param):
    global done, points, current, temp, polygons, polygon_selected, delete_mode, drawing_mode

    if delete_mode and event == cv2.EVENT_LBUTTONDOWN:
        # Check if clicked inside any polygon
        for i, poly in enumerate(polygons):
            if cv2.pointPolygonTest(poly, (x, y), False) >= 0:
                print(f"Polygon {i} selected for deletion.")
                polygon_selected = i
                break
        if polygon_selected != -1:
            # Delete the selected polygon
            del polygons[polygon_selected]
            polygon_selected = -1
            # Clear and update the image
            temp = clone.copy()
            for poly in polygons:
                cv2.polylines(temp, [poly], isClosed=True, color=(255, 0, 0), thickness=2)
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
            if len(points) > 2:
                print(f"Completing polygon with {len(points)} points.")
                polygons.append(np.array(points))  # Save current polygon
                points = []  # Reset points for a new polygon
                temp = img.copy()
            else:
                print("Need at least 3 points to complete a polygon")

cv2.namedWindow("image", cv2.WINDOW_NORMAL)

# Get screen size and resize the window to fit the screen
screen_width, screen_height = get_screen_size()
cv2.resizeWindow("image", screen_width, screen_height)

cv2.setMouseCallback("image", on_mouse)

while True:
    img = temp.copy()

    # Draw all the saved polygons (detected and newly drawn)
    for poly in polygons:
        poly = poly.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(img, [poly], isClosed=True, color=(255, 0, 0), thickness=2)

    # Draw the current polygon-in-progress
    if len(points) > 1:
        cv2.polylines(img, [np.array(points)], False, (0, 0, 255), 1)
        cv2.line(img, (points[-1][0], points[-1][1]), current, (0, 0, 255))

    # Update the window
    cv2.imshow("image", img)

    # Check for user input
    key = cv2.waitKey(50) & 0xFF

    if key == ord('d'):  # Press 'd' when done with all polygons
        print("Finalizing polygons...")
        break
    elif key == ord('n'):  # Press 'n' to start a new polygon
        if len(points) > 2:
            polygons.append(np.array(points))
            points = []
            temp = img.copy()
            print("Starting new polygon.")
        else:
            print("Need at least 3 points to complete a polygon.")
    elif key == ord('r'):  # Press 'r' to reset the current in-progress polygon
        show_popup("Try drawing again")
        points = []
        temp = clone.copy()
        print("Reset current polygon.")
    elif key == ord('b'):  # Press 'b' to enter delete mode
        show_popup("Delete polygon mode!")
        delete_mode = True
        drawing_mode = False
        print("Delete mode: click inside a polygon to delete it.")
    elif key == ord('e'):  # Press 'e' to exit delete mode
        show_popup("Exit delete mode")
        delete_mode = False
        drawing_mode = True
        print("Exit delete mode.")

# Final drawing of all polygons
for poly in polygons:
    poly = poly.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(final_img, [poly], isClosed=True, color=(255, 0, 0), thickness=2)

# Show final image
cv2.imshow("Polygon Drawer", final_img)
cv2.waitKey(0)

# Save the image with polygons
cv2.imwrite(main_folder_dir + "v2result/manual/mask-" + os.path.basename(path), final_img)
save_polygons_to_file(polygons,  main_folder_dir + "v2result/manual/polygons/" + roadNum + '.txt')
cv2.destroyAllWindows()

