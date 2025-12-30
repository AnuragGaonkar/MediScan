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
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(BASE_DIR, filename)

# ============================================================
# MEDICAL VALIDATION & HEURISTICS
# ============================================================

def is_likely_medical_image(image):
    if image is None: return False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    contrast = np.std(gray)
    if contrast < 35 or mean_intensity < 50 or mean_intensity > 210:
        return False
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([20, 255, 255]))
    skin_ratio = np.sum(skin_mask > 0) / (image.shape[0] * image.shape[1])
    return skin_ratio <= 0.25

def anatomical_region_hint(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    std = np.std(gray)
    _, bone = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    bone_ratio = np.sum(bone == 255) / (h * w)
    if bone_ratio > 0.18: return "HeadCT"
    dark_ratio = np.sum(gray < 60) / gray.size
    if dark_ratio > 0.40 and std < 70: return "CXR"
    if dark_ratio > 0.30 and std > 60: return "ChestCT"
    return "Unknown"

# ============================================================
# ML PREDICTION LOGIC (UPDATED FOR MLP)
# ============================================================

def predict_image_type(img_flattened):
    """
    Updated to use the Hidden Layer logic:
    Input -> (W1, b1) -> ReLU -> (W2, b2) -> Softmax
    """
    # 1. Load weights
    W1 = np.loadtxt(get_path("mlp_W1.csv"), delimiter=",")
    b1 = np.loadtxt(get_path("mlp_b1.csv"), delimiter=",")
    W2 = np.loadtxt(get_path("mlp_W2.csv"), delimiter=",")
    b2 = np.loadtxt(get_path("mlp_b2.csv"), delimiter=",")
    
    label_mapping_df = pd.read_csv(get_path("label_mapping.csv"))
    label_mapping = dict(zip(label_mapping_df["Index"], label_mapping_df["Label"]))

    # 2. Preprocess (matching your training script: scale to 0-1)
    img_processed = img_flattened / 255.0

    # 3. Forward Pass
    # Layer 1: Input to Hidden
    z1 = np.dot(img_processed, W1) + b1
    a1 = np.maximum(0, z1) # ReLU
    
    # Layer 2: Hidden to Output
    z2 = np.dot(a1, W2) + b2
    exp_z = np.exp(z2 - np.max(z2))
    probs = exp_z / np.sum(exp_z)
    
    predicted_class = int(np.argmax(probs))
    confidence = float(np.max(probs) * 100)
    predicted_type = label_mapping.get(predicted_class, "Unknown")

    return predicted_type, confidence

def predict_disease(image_type, img_flattened):
    # (Keeping your existing Softmax model configs for specific diseases)
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

    # Note: Using your original scaling for specific models
    if image_type == "ChestCT":
        img_processed = (img_flattened - np.mean(img_flattened)) / (np.std(img_flattened) + 1e-8)
    else:
        mean = np.loadtxt(get_path(cfg["mean"]), delimiter=",")
        std = np.loadtxt(get_path(cfg["std"]), delimiter=",")
        img_processed = (img_flattened - mean) / (np.maximum(std, 1e-8))

    z = np.dot(img_processed.reshape(1, -1), W) + b
    exp_z = np.exp(z - np.max(z))
    probs = exp_z / np.sum(exp_z)
    
    pred_class = int(np.argmax(probs))
    confidence = float(np.max(probs) * 100)
    label = cfg["labels"].get(pred_class, "Unknown")
    status = "Diseased" if label != "Normal" else "Healthy"
    return status, label, confidence

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

        if not is_likely_medical_image(image):
            return jsonify({
                "image_type": "Non-medical",
                "status": "Invalid",
                "disease": "Photo/Selfie detected – upload CT/MRI/X-ray only"
            }), 400

        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (64, 64))
        img_flat = img_resized.flatten()

        anatomy_hint = anatomical_region_hint(image)
        ml_type, ml_conf = predict_image_type(img_flat)

        final_type = ml_type
        final_type_conf = ml_conf
        if anatomy_hint != "Unknown" and anatomy_hint != ml_type:
            final_type = anatomy_hint
            final_type_conf = min(ml_conf, 85.0)

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
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)