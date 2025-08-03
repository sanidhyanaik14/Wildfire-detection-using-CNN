from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)


model = load_model("wildfire_model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        file_path = os.path.join('static', file.filename)
        file.save(file_path)

        
        img = image.load_img(file_path, target_size=(64, 64))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        
        prediction = model.predict(img_array)[0][0]
        result = "Wildfire" if prediction > 0.5 else "No Wildfire"

        return render_template('index.html', result=result, user_image=file_path)

if __name__ == '__main__':
    app.run(debug=True)
