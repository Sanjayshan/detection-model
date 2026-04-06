from flask import Flask, request, jsonify, send_from_directory, render_template
import logging
from flask_cors import CORS
from ultralytics import YOLO
import os
import uuid
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best.onnx')

app.logger.info(f"Looking for model at: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# ✅ Lazy load model (IMPORTANT)
model = None

def get_model():
    global model
    if model is None:
        try:
            app.logger.info("Loading YOLO model...")
            model = YOLO(MODEL_PATH)
            app.logger.info("Model loaded successfully ✅")
        except Exception as e:
            app.logger.error(f"Model loading failed: {e}")
            raise e
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
            return jsonify({"result": "No image uploaded"}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({"result": "No selected image"}), 400

        unique_name = str(uuid.uuid4()) + ".jpg"
        filepath = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(filepath)

        app.logger.info(f"Saved upload: {filepath}")

        # ✅ Load model only when needed
        model = get_model()

        # ✅ Resize image (prevents memory crash)
        img = cv2.imread(filepath)
        img = cv2.resize(img, (640, 640))

        results = model(img, verbose=False)
        app.logger.info("Inference complete")

        plotted_image = results[0].plot()

        result_filename = "result_" + unique_name
        result_path = os.path.join(RESULT_FOLDER, result_filename)

        saved = cv2.imwrite(result_path, plotted_image)
        if not saved:
            return jsonify({"result": "Failed to save result image"}), 500

        image_url = request.host_url + "results/" + result_filename

        labels = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = results[0].names[cls_id]
            conf = float(box.conf[0])
            labels.append({
                "label": label,
                "confidence": round(conf, 2)
            })

        return jsonify({
            "result": "Prediction completed",
            "image_url": image_url,
            "detections": labels
        })

    except Exception as e:
        import traceback
        traceback.print_exc()

        return jsonify({
            "result": "Server error",
            "error": str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # ✅ important for cloud
    app.run(host='0.0.0.0', port=port)