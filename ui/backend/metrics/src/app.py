"""
Metrics Microservice
SUMMARY: Computation of queue length and traffic density
"""
from flask import Flask, request, jsonify
import os
import requests
import socket
import datamall_credentials

# Enabling print statements in terminal
os.environ['PYTHONUNBUFFERED'] = '1'

app = Flask(__name__)

# Global Variable: Datamall API 
header = {"AccountKey": datamall_credentials.key,"accept":'application/json'}
url = 'http://datamall2.mytransport.sg/ltaodataservice/Traffic-Imagesv2'

# Checking if service is reachable
@app.route('/')
def default_check():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    return jsonify(
            {
                "message": "Service is reachable.",
                "service:": "metrics",
                "ip_address": local_ip
            }
    ), 200
    
# Checking if service is healthy
@app.route("/health")
def health_check():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    return jsonify(
            {
                "message": "Service is healthy.",
                "service:": "metrics",
                "ip_address": local_ip
            }
    ), 200

@app.route('/get_queue', methods=['POST'])
def get_queue():
    return

"""
TRAFFIC DENSITY POST REQUEST
INPUT:
    Type: JSON Data of an array of camera ids
    Example: {"id" : [2703, 9706]}

OUTPUT:
    Type: JSON
    Example: {2703: 'Low', 9706: 'Medium'}
"""
@app.route('/density', methods=['POST'])
def get_density():
    # Access JSON data from request body
    data = request.json  
    cam_id = data.get('id')
    
    # images return a dictionary of cameraid and its respective image link
    images = get_current_image(cam_id)
    # if there is a 400 error in get_current_image, return bad request code
    if type(images) == tuple:
        return images
    
    densities = {}
    try:
        for i in images.keys():
            image_bytes = images.get(i)
            density_result =  requests.post("http://127.0.0.1:3000/predict", files={"image" : image_bytes})
            densities.update({i: density_result.content.decode('utf-8')})
    except Exception as e:
        return jsonify(
        {
            'Internal Server Error': 'Density service call failed'
        }
    ), 500

    return jsonify(densities), 200


"""
TRAFFIC DENSITY FUNCTIONS
"""
# return a dictionary of camera id and its respective image link
def get_current_image(cam_id):
    # Call DataMall API
    response = requests.get(url, headers=header)
    API_Data = response.json() 
    try:
        image_detail = API_Data["value"]
        images = {}
        # Create a dictionary for faster lookups
        image_detail_dict = {j["CameraID"]: j for j in image_detail}
        for i in cam_id:
            id = str(i)
            if id in image_detail_dict:
                img_bytes = download(image_detail_dict[id]["ImageLink"], i)
                if type(img_bytes) == tuple:
                    return img_bytes
                images.update({i: img_bytes})

        if len(images) == 0:
            return jsonify(
            {
                'Error': 'No valid cameras given'
            }
        ), 400
        return images

    except Exception as e:
        return jsonify(
            {
                'Error': 'Image Dict failed'
            }
        ), 400

# get image bytes
def download(image_url, cameraID):  
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            idata = response.content
            return idata
        else:
            print("Error, retrying")
            response = requests.get(image_url)
            idata = response.content
            return idata

    except Exception as e:
        return jsonify(
            {
                'Error': 'Image Download Failed'
            }
        ), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3002)
