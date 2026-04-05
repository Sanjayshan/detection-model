from flask import Flask, request, jsonify, send_from_directory, render_template
import logging
from flask_cors import CORS
from ultralytics import YOLO
import os
import uuid
import cv2

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')

app.logger.info(f"Looking for model at: {MODEL_PATH}")
app.logger.info(f"Model exists: {os.path.exists(MODEL_PATH)}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = YOLO(MODEL_PATH)

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
        app.logger.info("Predict endpoint hit")

        if 'image' not in request.files:
            return jsonify({"result": "No image uploaded"}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({"result": "No selected image"}), 400

        unique_name = str(uuid.uuid4()) + ".jpg"
        filepath = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(filepath)
        app.logger.info(f"Image saved to: {filepath}")

        if not os.path.exists(filepath):
            return jsonify({"result": "Failed to save uploaded image"}), 500

        results = model(filepath)
        app.logger.info("Inference complete")

        plotted_image = results[0].plot()

        result_filename = "result_" + unique_name
        result_path = os.path.join(RESULT_FOLDER, result_filename)

        success = cv2.imwrite(result_path, plotted_image)
        if not success:
            app.logger.error(f"cv2.imwrite failed for path: {result_path}")
            return jsonify({"result": "Failed to save result image"}), 500

        app.logger.info(f"Result saved to: {result_path}")

        image_url = request.host_url + "results/" + result_filename

        labels = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = results[0].names[cls_id]
            conf = float(box.conf[0])
            labels.append({"label": label, "confidence": round(conf, 2)})

        return jsonify({
            "result": "Prediction completed",
            "image_url": image_url,
            "detections": labels
        })

    except Exception as e:
        app.logger.exception("Error during prediction")
        return jsonify({
            "result": f"Server error: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)