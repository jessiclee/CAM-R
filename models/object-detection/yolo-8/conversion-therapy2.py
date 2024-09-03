import os
custom_to_yolo_classses = {
    0:5,
    1:7,
    2:3,
    3:2
}

def convert_classes(path):
    for file in os.listdir(path):
        if file.endswith('.txt') and os.path.getsize(path + "/" + file) > 0:
            with open(path + "/" + file, 'r') as f:
                lines = f.readlines()
                modified_lines = []
                for line in lines:
                    parts = line.strip().split()  # Split line into part
                    if parts:
                        initial = parts[0]
                        # print("initial class and conversion class", initial, custom_to_yolo_classses.get(int(initial)))
                        parts[0] = str(custom_to_yolo_classses.get(int(initial)))
                        modified_lines.append(' '.join(parts))
            with open(path + "/" + file, 'w') as f:
                print("changed lines: ", modified_lines)
                f.write('\n'.join(modified_lines))

convert_classes("C:/Users/User/CAM-R/models/object-detection/yolo_data/test-coco-certified/labels")