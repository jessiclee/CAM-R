import os
import cv2
import xml.etree.ElementTree as ET

def yolo_to_voc_bbox(yolo_bbox, img_width, img_height):
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
    img_name = os.path.basename(image_path)

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = os.path.basename(os.path.dirname(image_path))
    ET.SubElement(annotation, "filename").text = img_name
    ET.SubElement(annotation, "path").text = image_path

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(img_width)
    ET.SubElement(size, "height").text = str(img_height)
    ET.SubElement(size, "depth").text = str(img_depth)

    for line in yolo_annotations:
        label, x_center, y_center, width, height = map(float, line.strip().split())
        xmin, ymin, xmax, ymax = yolo_to_voc_bbox((x_center, y_center, width, height), img_width, img_height)

        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = class_names[int(label)]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(xmin)
        ET.SubElement(bbox, "ymin").text = str(ymin)
        ET.SubElement(bbox, "xmax").text = str(xmax)
        ET.SubElement(bbox, "ymax").text = str(ymax)

    tree = ET.ElementTree(annotation)
    xml_filename = os.path.join(voc_save_path, img_name.replace('.jpg', '.xml'))
    tree.write(xml_filename)

def labels_to_xml(images_dir, yolo_annot_dir, voc_annot_dir, class_names):
    for txt_file in os.listdir(yolo_annot_dir):
        if txt_file.endswith('.txt'):
            with open(os.path.join(yolo_annot_dir, txt_file), 'r') as f:
                yolo_annotations = f.readlines()
            
            image_path = os.path.join(images_dir, txt_file.replace('.txt', '.jpg'))
            im = cv2.imread(image_path)
            h, w, c = im.shape
            create_voc_xml(image_path, yolo_annotations, voc_annot_dir, class_names, w, h, c)

if __name__ == '__main__':
    yolo_dir = "C:/Users/jesle/Desktop/fyp/actual_data/model_evaluation_data"
    voc_dir = "C:/Users/jesle/Desktop/fyp/actual_data/rcnn"
    os.makedirs(voc_dir, exist_ok=True)
    folders = os.listdir(yolo_dir)
    print(folders)
    class_names = ["bus", "truck", "motorcycle", "car"]
    for folder in folders:
        if folder in ['test', 'train', 'valid', 'test-coco-certified']:
            print("Processing folder:", folder)
            voc_pth = os.path.join(voc_dir, folder)
            os.makedirs(voc_pth, exist_ok=True)
            print("Saving annotations to:", voc_pth)
            if folder == 'test-coco-certified':
                labels_to_xml(yolo_dir + "/" + folder + "/images", yolo_dir + "/" + folder + "/labels", voc_pth, ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck'])
            else:
                labels_to_xml(yolo_dir + "/" + folder + "/images", yolo_dir + "/" + folder + "/labels", voc_pth, class_names)
        else:
            print("Skipping folder:", folder)
