# app.py
from flask import Flask, request, jsonify
from model import VGG16MultiClassClassifier
import socket
from io import BytesIO
from PIL import Image
import os

os.environ['PYTHONUNBUFFERED'] = '1'

app = Flask(__name__)
classifier = VGG16MultiClassClassifier(3)

@app.route('/')
def default_check():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    return jsonify(
            {
                "message": "Service is reachable.",
                "service:": "density",
                "ip_address": local_ip
            }
    ), 200
    
@app.route("/health")
def health_check():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    return jsonify(
            {
                "message": "Service is healthy.",
                "service:": "density",
                "ip_address": local_ip
            }
    ), 200

@app.route('/predict', methods=['POST'])
def predict():
    # # Get the image from the request
    # if 'image' not in request.files:
    #     return jsonify({'error': 'No image provided'}), 400
    # # Access the image file
    # image_file = request.files['image']
    # #     # Check if the file has a .jpeg extension
    # # if not image_file.filename.lower().endswith('.jpeg') or not image_file.filename.lower().endswith('.jpg') or not image_file.filename.lower().endswith('.png'):
    # #     return jsonify({'error': 'File must be a valid image type'}), 400
    # # Convert the image file to BytesIO
    # image_bytes = BytesIO(image_file.read())

    image_bytes = request.files['image']

    # Open the image using PIL
    image = Image.open(image_bytes)

    prediction = classifier.predictDensity(image, 'cpu')

    return prediction[1], 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)