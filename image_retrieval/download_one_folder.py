# import packages
import time
import os
from google.cloud import storage
from pathlib import Path
from os.path import isdir
from datetime import datetime, timedelta, timezone

storage_credentials = {
    1 : r'./fyp1433-google-storage.json',
    2 : r'./fyp2234-google-storage.json'
}

directory_structure = ["images/AYER RAJAH EXPRESSWAY/4701/",
                       "images/AYER RAJAH EXPRESSWAY/4702/",
                       "images/AYER RAJAH EXPRESSWAY/4704/",
                       "images/AYER RAJAH EXPRESSWAY/4705/",
                       "images/AYER RAJAH EXPRESSWAY/4706/",
                       "images/AYER RAJAH EXPRESSWAY/4707/",
                       "images/AYER RAJAH EXPRESSWAY/4708/",
                       "images/AYER RAJAH EXPRESSWAY/4709/",
                       "images/AYER RAJAH EXPRESSWAY/4710/",
                       "images/AYER RAJAH EXPRESSWAY/4712/",
                       "images/AYER RAJAH EXPRESSWAY/4714/",
                       "images/AYER RAJAH EXPRESSWAY/4716/",
                       "images/AYER RAJAH EXPRESSWAY/4798/",
                       "images/AYER RAJAH EXPRESSWAY/4799/",
                       "images/AYER RAJAH EXPRESSWAY/6716/",
                       "images/BUKIT TIMAH EXPRESSWAY/2703/",
                       "images/BUKIT TIMAH EXPRESSWAY/2704/",
                       "images/BUKIT TIMAH EXPRESSWAY/2705/",
                       "images/BUKIT TIMAH EXPRESSWAY/2706/",
                       "images/BUKIT TIMAH EXPRESSWAY/2707/",
                       "images/BUKIT TIMAH EXPRESSWAY/2708/",
                       "images/CENTRAL EXPRESSWAY/1701/",
                       "images/CENTRAL EXPRESSWAY/1702/",
                       "images/CENTRAL EXPRESSWAY/1703/",
                       "images/CENTRAL EXPRESSWAY/1704/",
                       "images/CENTRAL EXPRESSWAY/1705/",
                       "images/CENTRAL EXPRESSWAY/1706/",
                       "images/CENTRAL EXPRESSWAY/1707/",
                       "images/CENTRAL EXPRESSWAY/1709/",
                       "images/CENTRAL EXPRESSWAY/1711/",
                       "images/EAST COAST PARKWAY/1001/",
                       "images/EAST COAST PARKWAY/1113/",
                       "images/EAST COAST PARKWAY/3702/",
                       "images/EAST COAST PARKWAY/3705/",
                       "images/EAST COAST PARKWAY/3793/",
                       "images/EAST COAST PARKWAY/3795/",
                       "images/EAST COAST PARKWAY/3796/",
                       "images/EAST COAST PARKWAY/3797/",
                       "images/EAST COAST PARKWAY/3798/",
                       "images/KALLANG PAYA LEBAR EXPRESSWAY/1004/",
                       "images/KALLANG PAYA LEBAR EXPRESSWAY/1005/",
                       "images/KALLANG PAYA LEBAR EXPRESSWAY/1006/",
                       "images/KALLANG PAYA LEBAR EXPRESSWAY/1504/",
                       "images/KALLANG PAYA LEBAR EXPRESSWAY/3704/",
                       "images/KALLANG PAYA LEBAR EXPRESSWAY/5798/",
                       "images/KRANJI EXPRESSWAY/8701/",
                       "images/KRANJI EXPRESSWAY/8702/",
                       "images/KRANJI EXPRESSWAY/8704/",
                       "images/KRANJI EXPRESSWAY/8706/",
                       "images/MARINA COASTAL EXPRESSWAY/1501/",
                       "images/MARINA COASTAL EXPRESSWAY/1502/",
                       "images/MARINA COASTAL EXPRESSWAY/1503/",
                       "images/MARINA COASTAL EXPRESSWAY/1505/",
                       "images/PAN ISLAND EXPRESSWAY/1002/",
                       "images/PAN ISLAND EXPRESSWAY/1003/",
                       "images/PAN ISLAND EXPRESSWAY/5794/",
                       "images/PAN ISLAND EXPRESSWAY/5795/",
                       "images/PAN ISLAND EXPRESSWAY/5797/",
                       "images/PAN ISLAND EXPRESSWAY/5799/",
                       "images/PAN ISLAND EXPRESSWAY/6701/",
                       "images/PAN ISLAND EXPRESSWAY/6701/",
                       "images/PAN ISLAND EXPRESSWAY/6703/",
                       "images/PAN ISLAND EXPRESSWAY/6704/",
                       "images/PAN ISLAND EXPRESSWAY/6705/",
                       "images/PAN ISLAND EXPRESSWAY/6706/",
                       "images/PAN ISLAND EXPRESSWAY/6708/",
                       "images/PAN ISLAND EXPRESSWAY/6710/",
                       "images/PAN ISLAND EXPRESSWAY/6711/",
                       "images/PAN ISLAND EXPRESSWAY/6712/",
                       "images/PAN ISLAND EXPRESSWAY/6713/",
                       "images/PAN ISLAND EXPRESSWAY/6714/",
                       "images/PAN ISLAND EXPRESSWAY/6715/",
                       "images/SELETAR EXPRESSWAY/7797/",
                       "images/SELETAR EXPRESSWAY/7798/",
                       "images/SELETAR EXPRESSWAY/9701/",
                       "images/SELETAR EXPRESSWAY/9702/",
                       "images/SELETAR EXPRESSWAY/9703/",
                       "images/SELETAR EXPRESSWAY/9704/",
                       "images/SELETAR EXPRESSWAY/9705/",
                       "images/SELETAR EXPRESSWAY/9706/",
                       "images/TAMPINES EXPRESSWAY/1111/",
                       "images/TAMPINES EXPRESSWAY/1112/",
                       "images/TAMPINES EXPRESSWAY/7791/",
                       "images/TAMPINES EXPRESSWAY/7793/",
                       "images/TAMPINES EXPRESSWAY/7794/",
                       "images/TAMPINES EXPRESSWAY/7795/",
                       "images/TAMPINES EXPRESSWAY/7796/",
                       "images/TUAS CHECKPOINT/4703/",
                       "images/TUAS CHECKPOINT/4713/",
                       "images/WOODLANDS CHECKPOINT/2701/",
                       "images/WOODLANDS CHECKPOINT/2702/",
                       ]

def download_images(bucket_name,local_file_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r'./fyp1433-google-storage.json'
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        local_file_path = local_file_path.replace('\\', '/')
        local_file_path = local_file_path.replace('"', '')

        file_list = storage_client.list_blobs(bucket_name, prefix = 'images/SELETAR EXPRESSWAY/')
        file_list = [file.name for file in file_list]
        image_list = [file for file in file_list if file.endswith('.jpg')]

        for image in image_list:
            try:
                blob = bucket.blob(image)
                path = local_file_path + '/' + blob.name
                if isdir(path) == False:
                    make_directories(path)
                blob.download_to_filename(path) 
            except Exception as e:
                print("There is an error in downloading files")
                print(e)
        
    except Exception as e:
        print("There is an error in downloading files")
        print(e)

def make_directories(path):
    file_split = path.split("/")
    directory = "/".join(file_split[0:-1])
    Path(directory).mkdir(parents=True, exist_ok=True)

bucket_name = "fyp-lta-test"
local_file_path = "E:\images" ## PLACE FILE NAME HERE

download_images(bucket_name, local_file_path)