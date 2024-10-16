"""
Metrics Microservice
SUMMARY: Computation of queue length and traffic density
"""
from flask import Flask, request, jsonify
import os
import requests
import socket
import datamall_credentials
import pandas as pd
import queue_methods
import cv2
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd

# Enabling print statements in terminal
os.environ['PYTHONUNBUFFERED'] = '1'

app = Flask(__name__)

# Global Variable: Datamall API 
header = {"AccountKey": datamall_credentials.key,"accept":'application/json'}
url = 'http://datamall2.mytransport.sg/ltaodataservice/Traffic-Imagesv2'
color_dict = {
            1: (0, 0, 255), # red = truck
            2: (255, 0, 255), # pink = motorcycle
            3: (255,182,193), # grey ish = car
            4: (255, 255, 0) # neon blue = bus
        }
#Define vehicle properties 
vehicle_properties = {
    2: 1.5, 
    0: 1.5,
    1: 1.5,
    3: 1.5,
}

queue_threshold = 3
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

"""
TRAFFIC QUEUE POST REQUEST
INPUT:
    Type: JSON Data of an array of camera ids
    Example: {"id" : [2703, 9706]}
    
    or
    Type: Form Data
    Example: all fields + old image

OUTPUT:
    Type: JSON
    Example: {
            "queues": {
                "2703": {
                    "2": 1,
                    "4": 1
                }
            }
}
"""
@app.route('/get_queue', methods=['POST'])
def get_queue():
    cam_id = None
    images = None
    if 'image' in request.files:
        cam_id = request.form['id']
        old_image = request.files['image']
        old_image = old_image.read()
        images = {cam_id: old_image}
    else:
        data = request.json
        cam_id = data.get('id')
        # images return a dictionary of cameraid and its respective image link
        images = get_current_image(cam_id)
        # if there is a 400 error in get_current_image, return bad request code
        if type(images) == tuple:
            return images

    traffic_images = {}
    queues = {}
    try:
        for i in images.keys():
            image_bytes = images.get(i)
            road_info = requests.get("http://127.0.0.1:3001/lane/" + str(i)).json()
            width = road_info["data"]["width"]
            height = road_info["data"]["height"]
            lines = queue_methods.load_lanes(road_info["data"]["lanes"])
            try:
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                print(e)
            lane_id = 1
            for line in lines:
                midpoint_x = int((line[0][0] + line[1][0]) / 2)
                midpoint_y = int((line[0][1] + line[1][1]) / 2)
                try:
                    cv2.putText(image,str(lane_id), (midpoint_x,midpoint_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    line = line.reshape((-1, 1, 2)).astype(np.int32)
                    cv2.polylines(image, [line], isClosed=False, color=(255, 0, 0), thickness=2)
                except Exception as e:
                    print(e)
                lane_id += 1
            
            try:
                queue_result =  requests.post("http://127.0.0.1:3003/predict", files={"image" : image_bytes}).json()
                if len(queue_result) == 0:
                    return jsonify(
                        {
                            "Message": "no vehicles detected"
                        }
                    ), 200

                print("continue if not empty")
                queue_df = pd.DataFrame(queue_result)
                centroids = queue_methods.get_vehicles_from_df(queue_df)

                result_dict = {}
                for centroid in centroids:
                    coord = (centroid[0], centroid[1])
                    cv2.circle(image, coord, 5, (0, 200, 0), -1)
                    # This prints out the coordinates of the centroids on the image
                    cv2.putText(image,"(" + str(centroid[0]) +", " + str(centroid[1]) +")", (centroid[0] + 4, centroid[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    centroid_line_assignment, projection, isValid = queue_methods.line_assignment_by_perpendicular_distance(coord, lines)
                    if isValid is True:
                        cv2.circle(image, (int(projection[0]), int(projection[1])), 5, color_dict.get(centroid[6]), -1)
                        grouping = result_dict.get(centroid_line_assignment)
                        if grouping is None:
                            grouping = [centroid]
                        else:
                            grouping.append(centroid)
                        result_dict.update({centroid_line_assignment: grouping})
                    else:
                        cv2.circle(image, coord, 3, (0,0,0), -1)
                print("result_dict:")
                print(result_dict)
                lane_data = queue_methods.read_bboxes(result_dict)
                print(lane_data)
                road_lane_directions = requests.get("http://127.0.0.1:3001/direction/" + str(i)).json()
                print(road_lane_directions)
                queue_lengths= queue_methods.compute_queue_lengths(lane_data, image, vehicle_properties, queue_threshold, road_lane_directions, str(i))
                queues[i] = queue_lengths
                
                #save image locally
                cv2.imwrite("C:/Users/Zhiyi/Desktop/FYP/newtraffic/backend/" + str(i) + ".jpg", image)

                traffic_images[i] = image.tolist()
            except Exception as e:
                print(e)
            print(queues)

    except Exception as e:
        return jsonify(
        {
            'Internal Server Error': 'service call failed'
        }
    ), 500

    return jsonify({
                "queues" : queues
                }), 200

"""
TRAFFIC DENSITY POST REQUEST
INPUT:
    Type: JSON Data of an array of camera ids
    Example: {"id" : [2703, 9706]}
    
    or
    Type: Form Data
    Example: all fields + old image

OUTPUT:
    Type: JSON
    Example: {2703: 'Low', 9706: 'Medium'}
"""
@app.route('/density', methods=['POST'])
def get_density():
    cam_id = None
    images = None
    if 'image' in request.files:
        cam_id = request.form['id']
        old_image = request.files['image']
        old_image = old_image.read()
        images = {cam_id: old_image}
    else:
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
DATAMALL FUNCTIONS
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
    app.run(host='0.0.0.0', port=3002, debug = True)
