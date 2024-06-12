from flask import Flask, request, render_template
from flask_cors import CORS
from keras.models import load_model
import cv2
import numpy as np
from keras.applications.resnet50 import ResNet50
import base64

app = Flask(__name__)
CORS(app)

# Loading the fine-tuned ResNet model
model_path = 'models/resnet_finetuned_model.h5'
model = load_model(model_path)
map_dict = {0: 'leukemia', 1: 'myeloma', 2: 'normal'}

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route("/result", methods=['POST'])
def result():
    uploaded_file = request.files['inputImage']
    if uploaded_file.filename != '':
        file_extension = uploaded_file.filename.split('.')[-1]
        allowed_ext = ["jpg", "jpeg", "png", "bmp"]
        if file_extension.lower() in allowed_ext:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (128, 128))
            img = img / 255.0

            input_img = np.expand_dims(img, axis=0)
            
            prediction = np.argmax(model.predict(input_img), axis=-1)[0]

            image_data = base64.b64encode(file_bytes).decode('utf-8')
            return render_template('result.html', prediction=map_dict[prediction], image_data=image_data)
        else:
            return "Invalid file extension. Please upload an image with one of the following extensions: " + ', '.join(allowed_ext)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
