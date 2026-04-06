from flask import Flask, request, jsonify, send_from_directory, render_template
import logging
from flask_cors import CORS
from ultralytics import YOLO
import os
import uuid
import cv2

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best.onnx')

model = None

def get_model():
    global model
    if model is None:
        app.logger.info("Loading model...")
        model = YOLO(MODEL_PATH)
    return model


UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "results")

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
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']

        filename = str(uuid.uuid4()) + ".jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        model = get_model()

        # 🔥 Reduce memory usage
        img = cv2.imread(filepath)
        img = cv2.resize(img, (416, 416))

        results = model(img)

        plotted = results[0].plot()

        result_name = "res_" + filename
        result_path = os.path.join(RESULT_FOLDER, result_name)

        cv2.imwrite(result_path, plotted)

        return jsonify({
            "result": "success",
            "image_url": request.host_url + "results/" + result_name
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)