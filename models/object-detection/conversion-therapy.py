import os

official_classses = {
    "bus":0,
    "truck":1,
    "motorcycle":2,
    "car":3
}
filename = "obj.names"
cameras = os.listdir("./training")

for camera in cameras:
    #retrieve txt file first
    data_dict={}
    file_path= "./training/"+camera+"/"+filename
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            # content = file.read()
            # print(content)  # Print the file content
            for index, line in enumerate(file):
            # Strip whitespace from each line and add to dictionary
                item = line.strip()
                data_dict[index] = item  # Using 1-based indexing
        if(data_dict==official_classses):
            print(f"{camera} syncs")
        else:
            print(f"{camera} does not sync")
            print("current diction", data_dict)
            for file in os.listdir("./training/"+camera):
                file_path = os.path.join("./training/"+camera, file)
                if file.endswith('.txt') and file != "train.txt" and os.path.getsize(file_path) > 0:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        modified_lines = []
                        for line in lines:
                            parts = line.strip().split()  # Split line into part
                            if parts:
                                obj_class = data_dict.get(int(parts[0])) 
                                print("original class and number", parts[0], obj_class)
                                real_num = official_classses.get(obj_class)
                                print("actual num", real_num)
                                parts[0] = str(real_num)
                                modified_lines.append(' '.join(parts))
                    with open(file_path, 'w') as f:
                        print("changed lines: ", modified_lines)
                        f.write('\n'.join(modified_lines))

print("finished")