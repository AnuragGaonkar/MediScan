import os
import traceback
import numpy as np
import pandas as pd
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

# ============================================================
# APP SETUP & PATH HANDLING
# ============================================================
app = Flask(__name__)
# Allows all origins - essential for connecting frontend to Render
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(BASE_DIR, filename)

# ============================================================
# MEDICAL VALIDATION & ANATOMICAL HEURISTICS
# ============================================================

def is_likely_medical_image(image):
    if image is None: return False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    contrast = np.std(gray)

    # Basic threshold for valid medical imaging contrast
    if contrast < 35 or mean_intensity < 50 or mean_intensity > 210:
        return False

    # Simple skin detection to avoid selfies/photos
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([20, 255, 255]))
    skin_ratio = np.sum(skin_mask > 0) / (image.shape[0] * image.shape[1])
    
    return skin_ratio <= 0.25

def anatomical_region_hint(image):
    """Provides a rule-based backup for modality detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    std = np.std(gray)
    
    # Head CT → Skull ring detection (High intensity at edges)
    _, bone = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    bone_ratio = np.sum(bone == 255) / (h * w)
    if bone_ratio > 0.18: return "HeadCT"

    # CXR/Chest CT → Large dark lung areas
    dark_ratio = np.sum(gray < 60) / gray.size
    if dark_ratio > 0.40 and std < 70: return "CXR"
    if dark_ratio > 0.30 and std > 60: return "ChestCT"

    return "Unknown"

# ============================================================
# CORE ML UTILITIES
# ============================================================

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

def load_and_preprocess_image(image, image_size=(64, 64)):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, image_size)
    return img_resized.flatten()

def feature_scaling(image_data, mean, std):
    return (image_data - mean) / (np.maximum(std, 1e-8))

# ============================================================
# ML PREDICTION LOGIC
# ============================================================

def predict_image_type(img_flattened):
    W = np.loadtxt(get_path("softmax_weights.csv"), delimiter=",")
    b = np.loadtxt(get_path("softmax_bias.csv"), delimiter=",")
    X_mean = np.loadtxt(get_path("train_mean.csv"), delimiter=",")
    X_std = np.loadtxt(get_path("train_std.csv"), delimiter=",")
    
    label_mapping_df = pd.read_csv(get_path("label_mapping.csv"))
    label_mapping = dict(zip(label_mapping_df["Index"], label_mapping_df["Label"]))

    img_processed = feature_scaling(img_flattened, X_mean, X_std)
    z = np.dot(img_processed, W) + b
    probs = softmax(z)
    
    predicted_class = int(np.argmax(probs))
    confidence = float(np.max(probs) * 100)
    predicted_type = label_mapping.get(predicted_class, "Unknown")

    return predicted_type, confidence

def predict_disease(image_type, img_flattened):
    MODEL_CONFIGS = {
        "AbdomenCT": {
            "weights": "abdomen_softmax_weights.csv", "bias": "abdomen_softmax_bias.csv",
            "mean": "abdomen_train_mean.csv", "std": "abdomen_train_std.csv",
            "labels": {0: "Cyst", 1: "Normal", 2: "Stone", 3: "Tumor"}
        },
        "HeadCT": {
            "weights": "head_ct_softmax_weights.csv", "bias": "head_ct_softmax_bias.csv",
            "mean": "head_ct_train_mean.csv", "std": "head_ct_train_std.csv",
            "labels": {0: "Alzheimers-Mild", 1: "Alzheimers-Moderate", 2: "Alzheimers-Very Mild", 3: "Normal", 4: "Glioma", 5: "Meningioma", 6: "Pituitary"}
        },
        "CXR": {
            "weights": "cxr_softmax_weights.csv", "bias": "cxr_softmax_bias.csv",
            "mean": "cxr_train_mean.csv", "std": "cxr_train_std.csv",
            "labels": {0: "Normal", 1: "Pneumonia", 2: "Tuberculosis"}
        },
        "ChestCT": {
            "weights": "chest_softmax_weights.csv", "bias": "chest_softmax_bias.csv",
            "mean": "chest_train_mean.csv", "std": "chest_train_std.csv",
            "labels": {0: "Adenocarcinoma", 1: "Large Cell Carcinoma", 2: "Normal", 3: "Squamous Carcinoma"}
        },
        "BreastMRI": {
            "weights": "breast_softmax_weights.csv", "bias": "breast_softmax_bias.csv",
            "mean": "breast_train_mean.csv", "std": "breast_train_std.csv",
            "labels": {0: "Cancer", 1: "Normal"}
        }
    }

    if image_type not in MODEL_CONFIGS:
        return "Unknown", "Unknown", 0.0

    cfg = MODEL_CONFIGS[image_type]
    W = np.loadtxt(get_path(cfg["weights"]), delimiter=",")
    b = np.loadtxt(get_path(cfg["bias"]), delimiter=",").reshape(1, -1)

    if image_type == "ChestCT":
        img_processed = (img_flattened - np.mean(img_flattened)) / (np.std(img_flattened) + 1e-8)
    else:
        mean = np.loadtxt(get_path(cfg["mean"]), delimiter=",")
        std = np.loadtxt(get_path(cfg["std"]), delimiter=",")
        img_processed = feature_scaling(img_flattened, mean, std)

    z = np.dot(img_processed.reshape(1, -1), W) + b
    probs = softmax(z)
    
    pred_class = int(np.argmax(probs))
    confidence = float(np.max(probs) * 100)
    label = cfg["labels"].get(pred_class, "Unknown")

    status = "Diseased" if label != "Normal" else "Healthy"
    disease = label if status == "Diseased" else "Normal"

    return status, disease, confidence

# ============================================================
# API ENDPOINTS
# ============================================================

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "Server is running"}), 200

@app.route("/predict", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Pre-validation: Ensure it's likely a medical image
        if not is_likely_medical_image(image):
            return jsonify({
                "image_type": "Non-medical",
                "status": "Invalid",
                "disease": "Photo/Selfie detected – please upload a CT/MRI/X-ray"
            }), 400

        img_flat = load_and_preprocess_image(image)

        # Hybrid Modality Detection (ML + Anatomical Heuristic)
        anatomy_hint = anatomical_region_hint(image)
        ml_type, ml_conf = predict_image_type(img_flat)

        final_type = ml_type
        final_type_conf = ml_conf

        # If the ML contradicts the anatomy rules, prefer the hint for safety
        if anatomy_hint != "Unknown" and anatomy_hint != ml_type:
            final_type = anatomy_hint
            final_type_conf = min(ml_conf, 85.0)

        # Disease Prediction
        status, disease, disease_conf = predict_disease(final_type, img_flat)

        return jsonify({
            "image_type": final_type,
            "image_type_confidence": round(final_type_conf, 1),
            "status": status,
            "disease": disease,
            "disease_confidence": round(disease_conf, 1)
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

# ============================================================
# RENDER DEPLOYMENT SETTINGS
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)