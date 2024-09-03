import os
import shutil
import cv2
import xml.etree.ElementTree as ET

def yolo_to_voc_bbox(yolo_bbox, img_width, img_height):
    """
    Convert YOLO format bounding box to Pascal VOC format bounding box.
    """
    x_center, y_center, width, height = yolo_bbox
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    width_abs = width * img_width
    height_abs = height * img_height

    xmin = int(x_center_abs - (width_abs / 2))
    ymin = int(y_center_abs - (height_abs / 2))
    xmax = int(x_center_abs + (width_abs / 2))
    ymax = int(y_center_abs + (height_abs / 2))

    return xmin, ymin, xmax, ymax

def create_voc_xml(image_path, yolo_annotations, voc_save_path, class_names, img_width, img_height, img_depth):
    """
    Create a Pascal VOC XML file from YOLO annotations.
    """
    img_name = os.path.basename(image_path)

    # Create root element
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = os.path.basename(os.path.dirname(image_path))
    ET.SubElement(annotation, "filename").text = img_name
    ET.SubElement(annotation, "path").text = image_path

    # Source
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"

    # Size
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(img_width)
    ET.SubElement(size, "height").text = str(img_height)
    ET.SubElement(size, "depth").text = str(img_depth)

    for line in yolo_annotations:
        label, x_center, y_center, width, height = map(float, line.strip().split())
        xmin, ymin, xmax, ymax = yolo_to_voc_bbox((x_center, y_center, width, height), img_width, img_height)

        # Create object element
        obj = ET.SubElement(annotation, "object")
        print(label)
        ET.SubElement(obj, "name").text = class_names[int(label)]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(xmin)
        ET.SubElement(bbox, "ymin").text = str(ymin)
        ET.SubElement(bbox, "xmax").text = str(xmax)
        ET.SubElement(bbox, "ymax").text = str(ymax)

    # Save XML file
    tree = ET.ElementTree(annotation)
    xml_filename = os.path.join(voc_save_path, img_name.replace('.jpg', '.xml'))
    tree.write(xml_filename)

def labels_to_xml(images_dir, yolo_annot_dir, voc_annot_dir, class_names):
    for txt_file in os.listdir(yolo_annot_dir):
        if txt_file.endswith('.txt'):
            # Read YOLO annotations
            with open(os.path.join(yolo_annot_dir, txt_file), 'r') as f:
                yolo_annotations = f.readlines()
            
            # Image path
            image_path = os.path.join(images_dir, txt_file.replace('.txt', '.jpg'))
            im = cv2.imread(image_path)
            h, w, c = im.shape
            # Convert to VOC
            create_voc_xml(image_path, yolo_annotations, voc_annot_dir, class_names, w, h, c)

def transfer_imgs(src_folder, dest_folder, extensions=['.jpg', '.jpeg', '.png']):
    # Ensure the destination folder exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # Iterate over files in the source folder
    for file_name in os.listdir(src_folder):
        # Get the full path of the file
        file_path = os.path.join(src_folder, file_name)
        
        # Check if it's a file and has an image extension
        if os.path.isfile(file_path) and any(file_name.lower().endswith(ext) for ext in extensions):
            # Copy the file to the destination folder
            shutil.copy(file_path, dest_folder)
            print(f"Copied: {file_name}")

# Example usage
if __name__ == '__main__':
    yolo_dir = "C:/Users/jesle/Desktop/fyp/actual_data/model_evaluation_data"
    voc_dir = "C:/Users/jesle/Desktop/fyp/actual_data/rcnn"
    os.makedirs(voc_dir, exist_ok=True)
    folders = os.listdir(yolo_dir)
    print(folders)
    class_names = ["bus", "truck", "motorcycle", "car"]  # Replace with your class names
    for folder in folders:
        if folder == 'test' or folder == 'train' or folder =='valid' or folder == 'test-coco-certified':
            print("yes", folder)
            voc_pth = voc_dir + "/" + folder
            os.makedirs(voc_pth, exist_ok=True)
            print("test", voc_pth)
            #save all the images to voc
            transfer_imgs(yolo_dir + "/" + folder + "/images", voc_pth)
            if folder == 'test-coco-certified':
                labels_to_xml(voc_pth, yolo_dir + "/" + folder + "/labels", voc_pth, ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck'])
            else:
                labels_to_xml(voc_pth, yolo_dir + "/" + folder + "/labels", voc_pth, class_names)
        else:
            print("no", folder)