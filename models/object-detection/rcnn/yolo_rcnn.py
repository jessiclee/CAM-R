import os
from PIL import Image

def convert_yolo_to_custom(yolo_annotation_dir, output_annotation_dir, image_dir):
    if not os.path.exists(output_annotation_dir):
        os.makedirs(output_annotation_dir)
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        annotation_file = os.path.splitext(image_file)[0] + '.txt'
        yolo_annotation_path = os.path.join(yolo_annotation_dir, annotation_file)
        
        if not os.path.isfile(yolo_annotation_path):
            print(f"Annotation file not found for image: {image_file}")
            continue
        
        with open(yolo_annotation_path, 'r') as f:
            yolo_boxes = f.readlines()
        
        image = Image.open(image_path)
        width, height = image.size
        
        output_annotation_path = os.path.join(output_annotation_dir, annotation_file)
        with open(output_annotation_path, 'w') as f:
            for box in yolo_boxes:
                parts = box.strip().split()
                class_id = parts[0]
                x_center = float(parts[1])
                y_center = float(parts[2])
                box_width = float(parts[3])
                box_height = float(parts[4])
                
                xmin = int((x_center - box_width / 2) * width)
                xmax = int((x_center + box_width / 2) * width)
                ymin = int((y_center - box_height / 2) * height)
                ymax = int((y_center + box_height / 2) * height)
                
                f.write(f"{xmin},{ymin},{xmax},{ymax},{class_id}\n")

# Set the paths
yolo_annotation_dir = 'C:/Users/jesle/Desktop/fyp/actual_data/model_evaluation_data/test/labels'
output_annotation_dir = 'C:/Users/jesle/Desktop/fyp/actual_data/text_labels/test'
image_dir = 'C:/Users/jesle/Desktop/fyp/actual_data/model_evaluation_data/test/images'


convert_yolo_to_custom(yolo_annotation_dir, output_annotation_dir, image_dir)
