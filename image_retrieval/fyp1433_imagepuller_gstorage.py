import datamall_credentials
import requests
import time
import os
from io import BytesIO
from datetime import datetime
from google.cloud import storage

# Google Cloud Necessities

############ GLOBAL VARIABLES ################
header = {"AccountKey": datamall_credentials.key,"accept":'application/json'}
url = 'http://datamall2.mytransport.sg/ltaodataservice/Traffic-Imagesv2'
response = requests.get(url, headers=header)
API_Data = response.json() 
len = len(API_Data["value"])

CAMERAIDS = {
    "1001": "EAST COAST PARKWAY",
    "1002": "PAN ISLAND EXPRESSWAY",
    "1003": "PAN ISLAND EXPRESSWAY",
    "1004": "KALLANG PAYA LEBAR EXPRESSWAY",
    "1005": "KALLANG PAYA LEBAR EXPRESSWAY",
    "1006": "KALLANG PAYA LEBAR EXPRESSWAY",
    "1111": "TAMPINES EXPRESSWAY",
    "1112": "TAMPINES EXPRESSWAY",
    "1113": "EAST COAST PARKWAY",
    "1501": "MARINA COASTAL EXPRESSWAY",
    "1502": "MARINA COASTAL EXPRESSWAY",
    "1503": "MARINA COASTAL EXPRESSWAY",
    "1504": "KALLANG PAYA LEBAR EXPRESSWAY",
    "1505": "MARINA COASTAL EXPRESSWAY",
    "1701": "CENTRAL EXPRESSWAY",
    "1702": "CENTRAL EXPRESSWAY",
    "1703": "CENTRAL EXPRESSWAY",
    "1704": "CENTRAL EXPRESSWAY",
    "1705": "CENTRAL EXPRESSWAY",
    "1706": "CENTRAL EXPRESSWAY",
    "1707": "CENTRAL EXPRESSWAY",
    "1709": "CENTRAL EXPRESSWAY",
    "1711": "CENTRAL EXPRESSWAY",
    "2701": "WOODLANDS CHECKPOINT",
    "2702": "WOODLANDS CHECKPOINT",
    "2703": "BUKIT TIMAH EXPRESSWAY",
    "2704": "BUKIT TIMAH EXPRESSWAY",
    "2705": "BUKIT TIMAH EXPRESSWAY",
    "2706": "BUKIT TIMAH EXPRESSWAY",
    "2707": "BUKIT TIMAH EXPRESSWAY",
    "2708": "BUKIT TIMAH EXPRESSWAY",
    "3702": "EAST COAST PARKWAY",
    "3704": "KALLANG PAYA LEBAR EXPRESSWAY",
    "3705": "EAST COAST PARKWAY",
    "3793": "EAST COAST PARKWAY",
    "3795": "EAST COAST PARKWAY",
    "3796": "EAST COAST PARKWAY",
    "3797": "EAST COAST PARKWAY",
    "3798": "EAST COAST PARKWAY",
    "4701": "AYER RAJAH EXPRESSWAY",
    "4702": "AYER RAJAH EXPRESSWAY",
    "4703": "TUAS CHECKPOINT",
    "4704": "AYER RAJAH EXPRESSWAY",
    "4705": "AYER RAJAH EXPRESSWAY",
    "4706": "AYER RAJAH EXPRESSWAY",
    "4707": "AYER RAJAH EXPRESSWAY",
    "4708": "AYER RAJAH EXPRESSWAY",
    "4709": "AYER RAJAH EXPRESSWAY",
    "4710": "AYER RAJAH EXPRESSWAY",
    "4712": "AYER RAJAH EXPRESSWAY",
    "4713": "TUAS CHECKPOINT",
    "4714": "AYER RAJAH EXPRESSWAY",
    "4716": "AYER RAJAH EXPRESSWAY",
    "4798": "AYER RAJAH EXPRESSWAY",
    "4799": "AYER RAJAH EXPRESSWAY",
    "5794": "PAN ISLAND EXPRESSWAY",
    "5795": "PAN ISLAND EXPRESSWAY",
    "5797": "PAN ISLAND EXPRESSWAY",
    "5798": "KALLANG PAYA LEBAR EXPRESSWAY",
    "5799": "PAN ISLAND EXPRESSWAY",
    "6701": "PAN ISLAND EXPRESSWAY",
    "6703": "PAN ISLAND EXPRESSWAY",
    "6704": "PAN ISLAND EXPRESSWAY",
    "6705": "PAN ISLAND EXPRESSWAY",
    "6706": "PAN ISLAND EXPRESSWAY",
    "6708": "PAN ISLAND EXPRESSWAY",
    "6710": "PAN ISLAND EXPRESSWAY",
    "6711": "PAN ISLAND EXPRESSWAY",
    "6712": "PAN ISLAND EXPRESSWAY",
    "6713": "PAN ISLAND EXPRESSWAY",
    "6714": "PAN ISLAND EXPRESSWAY",
    "6715": "PAN ISLAND EXPRESSWAY",
    "6716": "AYER RAJAH EXPRESSWAY",
    "7791": "TAMPINES EXPRESSWAY",
    "7793": "TAMPINES EXPRESSWAY",
    "7794": "TAMPINES EXPRESSWAY",
    "7795": "TAMPINES EXPRESSWAY",
    "7796": "TAMPINES EXPRESSWAY",
    "7797": "SELETAR EXPRESSWAY",
    "7798": "SELETAR EXPRESSWAY",
    "8701": "KRANJI EXPRESSWAY",
    "8702": "KRANJI EXPRESSWAY",
    "8704": "KRANJI EXPRESSWAY",
    "8706": "KRANJI EXPRESSWAY",
    "9701": "SELETAR EXPRESSWAY",
    "9702": "SELETAR EXPRESSWAY",
    "9703": "SELETAR EXPRESSWAY",
    "9704": "SELETAR EXPRESSWAY",
    "9705": "SELETAR EXPRESSWAY",
    "9706": "SELETAR EXPRESSWAY",
}

# Dictionary to store image stream from download
image_data = {}
download_error = 0
# Function to download image from Datamall
def download(image_url, cameraID):  
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            idata = response.content
        else:
            print("Error, retrying")
            response = requests.get(image_url)
            idata = response.content
    except Exception as e:
        print(e)
        download_error += 1
    image_data[cameraID] = idata


def upload(timestamp):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r'./fyp1433-google-storage.json'
    bucket = ''
    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket("fyp-lta-test")
    except:
        print("bucket not found")
        return # In the case of bucket problems, type appropriate error handling
    blob_dictionary = {}
    
    for cid in image_data:
        image_name = cid + timestamp + ".jpg"
        filepath = "images/" + CAMERAIDS[cid] + "/" + cid + "/" + image_name
        blob_dictionary[cid] = bucket.blob(filepath)
        
    try:
        for key in blob_dictionary:
            blob = blob_dictionary[key]
            image = image_data[key]
            # default retry policy
            blob.upload_from_string(image, content_type='image/jpg')  
    except Exception as e:
        print("Something went wrong")
        print(e)
    return

############ main ################
def main():
    start_download = time.time()
    timestamp = datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
    # Downloading of all images from the Datamall API
    for i in range(len):
        try:
            image_detail = API_Data["value"][i]
            imgURL = image_detail["ImageLink"] 
            cameraID = image_detail["CameraID"] 

            # download all bytes
            download(imgURL, cameraID)
        
        except Exception as e:
            print(e + "cameraid")
    print("Number of times there were downloading errors: " + str(download_error))
    print("--- %s seconds to download images ---" % (time.time() - start_download))
            
    # Uploads all images to the Google Drive
    upload(timestamp)
    
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
