# app.py
from flask import Flask, request, jsonify
from model import YOLO_NAS_L
from io import BytesIO
from PIL import Image
import os
os.environ['PYTHONUNBUFFERED'] = '1'
import pandas as pd

app = Flask(__name__)
model = YOLO_NAS_L('19.pth')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Access the image file
    image_file = request.files['image']
    
    #     # Check if the file has a .jpeg extension
    # if not image_file.filename.lower().endswith('.jpeg') or not image_file.filename.lower().endswith('.jpg') or not image_file.filename.lower().endswith('.png'):
    #     return jsonify({'error': 'File must be a valid image type'}), 400

    # Convert the image file to BytesIO
    image_bytes = BytesIO(image_file.read())

    # Open the image using PIL
    image = Image.open(image_bytes)
    
    prediction = model.predict(image)
    return prediction.to_json(orient='records')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)

"""
response = requests.post(
    'http://localhost:5000/predict',
    files={'image': open(image_path, 'rb')}
)
"""