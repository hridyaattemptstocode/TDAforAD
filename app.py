from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np

from keras.models import load_model
from keras.preprocessing import image

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import traceback

app = Flask(__name__)

MODEL_PATH = 'deadass_cnn.h5'

# Load your trained model
model = None
try:
    model = load_model(MODEL_PATH)
    print('Model loaded successfully.')
except Exception as e:
    print('Error loading model:', e)

UPLOAD_FOLDER = 'images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Directory '{UPLOAD_FOLDER}' created.")

def model_predict(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(128, 128))

        # Preprocessing the image
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # Normalize pixel values (assuming pixels are in the range [0, 255])

        preds = model.predict(x)
        return preds
    except Exception as e:
        print('Error predicting:', e)
        traceback.print_exc()  # Print traceback for detailed error information
        return None

# Define a dictionary to map class indices to class names
class_map = {
    0: 'AD',
    1: 'CI',
    2: 'CN'
    # Add more class mappings as needed
}

# @app.route('/')
@app.route('/index', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        print('No file part')
        return 'No file uploaded'
    
    file = request.files['file']
    if file.filename == '':
        print('No selected file')
        return 'No selected file'

    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        if preds is not None:
            # Get the index of the maximum prediction
            pred_class_index = np.argmax(preds)
            # Map the index to the corresponding class name
            pred_class_name = class_map.get(pred_class_index, 'Unknown')
            # Return the class name as the result
            return pred_class_name
        else:
            return 'Error processing image'
    except Exception as e:
        print('Error uploading and predicting:', e)
        traceback.print_exc()  # Print traceback for detailed error information
        return 'Error processing request'

if __name__ == '__main__':
    app.run(port=5001, debug=True)