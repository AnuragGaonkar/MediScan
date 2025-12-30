from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def path(filename): return os.path.join(BASE_DIR, filename)

app = Flask(__name__)
CORS(app)

def is_likely_medical_image(image):
    if image is None: return False
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    contrast = np.std(gray)
    
    if contrast < 40 or mean_intensity < 60 or mean_intensity > 205:
        return False
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(hsv, np.array([0, 25, 50]), np.array([22, 255, 255]))
    skin_ratio = np.sum(skin_mask > 0) / (image.shape[0] * image.shape[1])
    
    if skin_ratio > 0.25:  # Selfies rejected
        return False
    
    return True

# YOUR EXACT OG FUNCTIONS (unchanged)
def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

def load_and_preprocess_image(image, image_size=(64, 64)):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, image_size)
    img_flattened = img_resized.flatten()
    return img_flattened

def feature_scaling(image_data, mean, std):
    return (image_data - mean) / (std + 1e-8)

def predict_image_type(img_flattened):
    W = np.loadtxt(path("softmax_weights.csv"), delimiter=",")
    b = np.loadtxt(path("softmax_bias.csv"), delimiter=",")
    X_mean = np.loadtxt(path("train_mean.csv"), delimiter=",")
    X_std = np.loadtxt(path("train_std.csv"), delimiter=",")
    label_mapping_df = pd.read_csv(path("label_mapping.csv"))
    label_mapping = dict(zip(label_mapping_df["Index"], label_mapping_df["Label"]))

    img_processed = feature_scaling(img_flattened, X_mean, X_std)
    z = np.dot(img_processed, W) + b
    y_pred = softmax(z)
    predicted_class = int(np.argmax(y_pred))
    predicted_type = label_mapping.get(predicted_class, "Unknown")

    return predicted_type, img_flattened

# [Keep predict_disease EXACTLY as your OG code]

@app.route("/predict", methods=["POST"])
def upload():
    file = request.files["file"]
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    # 🔥 NEW VALIDATION (only change)
    if not is_likely_medical_image(image):
        return jsonify({
            "error": "Not a medical image",
            "message": "Please upload CT/MRI/X-Ray scans only.",
            "image_type": "Non-medical",
            "status": "Invalid",
            "disease": "N/A"
        }), 400

    # YOUR OG PIPELINE (unchanged)
    img_flattened = load_and_preprocess_image(image)
    predicted_type, img_flattened = predict_image_type(img_flattened)
    disease_status, disease_type = predict_disease(predicted_type, img_flattened)

    response = {
        "image_type": predicted_type,
        "status": disease_status,
        "disease": disease_type
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
