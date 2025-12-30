from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import cv2
import os
import traceback

# ============================================================
# BASIC APP SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def path(filename):
    return os.path.join(BASE_DIR, filename)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ============================================================
# SAFETY PARAMETERS (CORE FIX)
# ============================================================

MIN_CONFIDENCE = 88.0

IMPOSSIBLE_PAIRS = {
    "BreastMRI": ["AbdomenCT", "ChestCT", "CXR"],
    "CXR": ["AbdomenCT", "HeadCT", "BreastMRI"],
    "HeadCT": ["CXR", "BreastMRI"],
}

# ============================================================
# MEDICAL IMAGE VALIDATION (UNCHANGED)
# ============================================================

def is_likely_medical_image(image):
    if image is None:
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    contrast = np.std(gray)

    if contrast < 35 or mean_intensity < 50 or mean_intensity > 210:
        return False

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([20, 255, 255]))
    skin_ratio = np.sum(skin_mask > 0) / (image.shape[0] * image.shape[1])

    return skin_ratio <= 0.25

# ============================================================
# ANATOMICAL REGION HINT (WEAK, NON-DESTRUCTIVE)
# ============================================================

def anatomical_region_hint(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mean = np.mean(gray)
    std = np.std(gray)

    _, bone = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    bone_ratio = np.sum(bone == 255) / (h * w)
    if bone_ratio > 0.18:
        return "HeadCT"

    dark_ratio = np.sum(gray < 60) / gray.size
    if dark_ratio > 0.45:
        return "CXR"

    if dark_ratio > 0.30 and std > 60:
        return "ChestCT"

    center = gray[h//3:2*h//3, w//3:2*w//3]
    if 25 < np.std(center) < 60:
        return "AbdomenCT"

    if std < 40 and mean > 90:
        return "BreastMRI"

    return "Unknown"

# ============================================================
# CORE ML UTILITIES
# ============================================================

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

def load_and_preprocess_image(image, image_size=(64, 64)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, image_size)
    return resized.flatten()

def feature_scaling(x, mean, std):
    return (x - mean) / (np.maximum(std, 1e-8))

# ============================================================
# IMAGE TYPE CLASSIFICATION (UNCHANGED MODEL)
# ============================================================

def predict_image_type(img_flat):
    W = np.loadtxt(path("softmax_weights.csv"), delimiter=",")
    b = np.loadtxt(path("softmax_bias.csv"), delimiter=",")
    X_mean = np.loadtxt(path("train_mean.csv"), delimiter=",")
    X_std = np.loadtxt(path("train_std.csv"), delimiter=",")
    label_df = pd.read_csv(path("label_mapping.csv"))
    label_map = dict(zip(label_df["Index"], label_df["Label"]))

    img_scaled = feature_scaling(img_flat, X_mean, X_std)
    z = np.dot(img_scaled, W) + b
    probs = softmax(z)

    cls = int(np.argmax(probs))
    conf = float(np.max(probs) * 100)

    return label_map.get(cls, "Unknown"), conf

# ============================================================
# DISEASE PREDICTION (UNCHANGED MODELS)
# ============================================================

def predict_disease(image_type, img_flat):

    MODEL_CONFIGS = {
        "AbdomenCT": {
            "weights": "abdomen_softmax_weights.csv",
            "bias": "abdomen_softmax_bias.csv",
            "mean": "abdomen_train_mean.csv",
            "std": "abdomen_train_std.csv",
            "labels": {0: "Cyst", 1: "Normal", 2: "Stone", 3: "Tumor"}
        },
        "HeadCT": {
            "weights": "head_ct_softmax_weights.csv",
            "bias": "head_ct_softmax_bias.csv",
            "mean": "head_ct_train_mean.csv",
            "std": "head_ct_train_std.csv",
            "labels": {
                0: "Alzheimers-Mild Dementia",
                1: "Alzheimers-Moderate Dementia",
                2: "Alzheimers-Very Mild Dementia",
                3: "Normal",
                4: "Tumor-glioma",
                5: "Tumor-meningioma",
                6: "Tumor-pituitary"
            }
        },
        "CXR": {
            "weights": "cxr_softmax_weights.csv",
            "bias": "cxr_softmax_bias.csv",
            "mean": "cxr_train_mean.csv",
            "std": "cxr_train_std.csv",
            "labels": {0: "Normal", 1: "Pneumonia", 2: "Tuberculosis"}
        },
        "ChestCT": {
            "weights": "chest_softmax_weights.csv",
            "bias": "chest_softmax_bias.csv",
            "mean": "chest_train_mean.csv",
            "std": "chest_train_std.csv",
            "labels": {
                0: "Adenocarcinoma LLL T2",
                1: "Large Cell Carcinoma LHL T2",
                2: "Normal",
                3: "Squamous Carcinoma LHL T1"
            }
        },
        "BreastMRI": {
            "weights": "breast_softmax_weights.csv",
            "bias": "breast_softmax_bias.csv",
            "mean": "breast_train_mean.csv",
            "std": "breast_train_std.csv",
            "labels": {0: "Cancer", 1: "Normal"}
        }
    }

    if image_type not in MODEL_CONFIGS:
        return "Unknown", "Unknown", 0.0

    cfg = MODEL_CONFIGS[image_type]
    W = np.loadtxt(path(cfg["weights"]), delimiter=",")
    b = np.loadtxt(path(cfg["bias"]), delimiter=",").reshape(1, -1)
    mean = np.loadtxt(path(cfg["mean"]), delimiter=",")
    std = np.loadtxt(path(cfg["std"]), delimiter=",")

    img_processed = feature_scaling(img_flat, mean, std)
    z = np.dot(img_processed.reshape(1, -1), W) + b
    probs = softmax(z)

    cls = int(np.argmax(probs))
    conf = float(np.max(probs) * 100)
    label = cfg["labels"].get(cls, "Unknown")

    status = "Diseased" if label != "Normal" else "Healthy"
    disease = label if status == "Diseased" else "Normal"

    return status, disease, conf

# ============================================================
# API ROUTES
# ============================================================

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "Server running ✅"})

@app.route("/predict", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image"}), 400

        if not is_likely_medical_image(image):
            return jsonify({
                "image_type": "Non-medical",
                "status": "Invalid",
                "disease": "Photo/Selfie detected – upload CT/MRI/X-ray only"
            }), 400

        img_flat = load_and_preprocess_image(image)

        anatomy_hint = anatomical_region_hint(image)
        ml_type, ml_conf = predict_image_type(img_flat)

        FINAL_TYPE = ml_type
        FINAL_CONF = ml_conf

        if anatomy_hint != "Unknown":
            blocked = IMPOSSIBLE_PAIRS.get(anatomy_hint, [])
            if ml_type in blocked:
                FINAL_TYPE = anatomy_hint
                FINAL_CONF = min(ml_conf, 85.0)

        # -------- CONFIDENCE-AWARE REJECTION --------
        if FINAL_CONF < MIN_CONFIDENCE:
            return jsonify({
                "image_type": "Uncertain",
                "image_type_confidence": round(FINAL_CONF, 1),
                "status": "Review Required",
                "disease": "Low confidence – manual review suggested",
                "disease_confidence": 0.0
            })

        status, disease, disease_conf = predict_disease(FINAL_TYPE, img_flat)

        return jsonify({
            "image_type": FINAL_TYPE,
            "image_type_confidence": round(FINAL_CONF, 1),
            "status": status,
            "disease": disease,
            "disease_confidence": round(disease_conf, 1)
        })

    except Exception:
        print(traceback.format_exc())
        return jsonify({"error": "Server error"}), 500

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
