#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 08:01:20 2023

@author: princesingh
"""

import cv2
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('/Users/princesingh/Desktop/road proj/model.h5')
# Define the class names
class_names = ('clean', 'dirty')

# Define the image size
img_size = (128, 128, 3)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    image_file = request.files['image']
    
    # Read and preprocess the image
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    resized_image = cv2.resize(image, img_size[0:2])[:, :, ::-1]
    processed_image = np.asarray(resized_image) / 255.0
    
    # Make the prediction
    prediction = model.predict(np.expand_dims(processed_image, axis=0))
    predicted_class = class_names[np.argmax(prediction)]
    
    # Return the result as JSON
    result = {'prediction': predicted_class}
    return jsonify(result)

if __name__ == '__main__':
    app.run()

