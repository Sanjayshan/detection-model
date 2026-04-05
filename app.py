from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from ultralytics import YOLO
import os
import uuid
import cv2

app = Flask(__name__)
CORS(app)

model = YOLO('best.pt')

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/results/<filename>')
def get_result_image(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"result": "No image uploaded"})

    file = request.files['image']

    unique_name = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(filepath)

    results = model(filepath)

    plotted_image = results[0].plot()

    result_filename = "result_" + unique_name
    result_path = os.path.join(RESULT_FOLDER, result_filename)

    cv2.imwrite(result_path, plotted_image)

    image_url = request.host_url + "results/" + result_filename

    return jsonify({
        "result": "Prediction completed",
        "image_url": image_url
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)