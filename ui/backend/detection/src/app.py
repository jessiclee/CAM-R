# app.py
from flask import Flask, request, jsonify
from model import YOLO_NAS_L
from io import BytesIO
from PIL import Image
import socket
import os
os.environ['PYTHONUNBUFFERED'] = '1'
import pandas as pd

app = Flask(__name__)
model = YOLO_NAS_L('19.pth')

# Checking if service is healthy
@app.route("/health")
def health_check():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    return jsonify(
            {
                "message": "Service is healthy.",
                "service:": "detection",
                "ip_address": local_ip
            }
    ), 200

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Access the image file
    image_bytes = request.files['image']
    # Open the image using PIL
    image = Image.open(image_bytes)

    prediction = model.predict(image)
    return prediction.to_json(orient='records'), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3003)
