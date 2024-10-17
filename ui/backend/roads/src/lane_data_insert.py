import os
import requests
import pandas as pd
import numpy as np


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

def send_req(camera, road, width, height, longi, lat, lanes):
    return requests.post("http://localhost:3001/lane", data = {
            "camera": camera,
            "road": road,
            "width": width,
            "height": height,
            "longi": longi,
            "lat": lat
        }, files = {
            "lanes": lanes
            }).json()
    

lane_path = "C:/Users/Zhiyi/Desktop/FYP/newtraffic/v3result/manual2/lines/"
info_path = "C:/Users/Zhiyi/Desktop/FYP/TrafficImages.csv"

df = pd.read_csv(info_path)

for index, row in df.iterrows():
    camera = int(row['CAMERA_ID'])
    road = row['LOCATION'] 
    width, height = get_camera_size(camera)
    longi = row['LONGITUDE']
    lat = row['LATITUDE']
    lane_file_path = lane_path + str(camera) + ".txt"
    with open(lane_file_path, 'rb') as file:
        lanes = file.read()
    print(camera)
    print(send_req(camera, road, width, height, longi, lat, lanes))