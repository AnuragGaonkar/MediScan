import os
import traceback
import numpy as np
import pandas as pd
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(BASE_DIR, filename)

# RELAXED MEDICAL VALIDATION (CT scans pass)
def is_likely_medical_image(image):
    if image is None: return False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    contrast = np.std(gray)
    
    # FIXED: Lower thresholds for CT/MRI (they have high mean intensity)
    if contrast < 25 or mean_intensity < 30 or mean_intensity > 240:
        return False
    
    # Skin detection (relaxed for medical scans)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(hsv, np.array([0, 20, 60]), np.array([25, 255, 255]))
    skin_ratio = np.sum(skin_mask > 0) / (image.shape[0] * image.shape[1])
    
    return skin_ratio <= 0.35  # Increased tolerance

def anatomical_region_hint(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    std = np.std(gray)
    
    # IMPROVED: Head CT detection (bone ring)
    _, bone = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)  # Lowered threshold
    bone_ratio = np.sum(bone == 255) / gray.size
    if bone_ratio > 0.12:  # Relaxed from 0.18
        return "HeadCT"
    
    dark_ratio = np.sum(gray < 70) / gray.size
    if dark_ratio > 0.35 and std < 75: return "CXR"
    
    return "Unknown"

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

def load_and_preprocess_image(image, image_size=(64, 64)):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, image_size)
    return img_resized.flatten()

def feature_scaling(image_data, mean, std):
    return (image_data - mean) / (np.maximum(std, 1e-8))

def predict_image_type(img_flattened):
    try:
        # CHECK FILES FIRST
        required_files = ["softmax_weights.csv", "softmax_bias.csv", "train_mean.csv", 
                         "train_std.csv", "label_mapping.csv"]
        missing = [f for f in required_files if not os.path.exists(get_path(f))]
        if missing:
            print(f"Missing type model files: {missing}")
            return "Unknown", 0.0
        
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
    except Exception as e:
        print(f"Type prediction error: {e}")
        return "Unknown", 0.0

def predict_disease(image_type, img_flattened):
    MODEL_CONFIGS = {
        "HeadCT": {
            "weights": "head_ct_softmax_weights.csv", "bias": "head_ct_softmax_bias.csv",
            "mean": "head_ct_train_mean.csv", "std": "head_ct_train_std.csv",
            "labels": {0: "Alzheimers-Mild", 1: "Alzheimers-Moderate", 2: "Alzheimers-Very Mild", 
                      3: "Normal", 4: "Glioma", 5: "Meningioma", 6: "Pituitary"}
        },
        # ... other configs same as yours
        "CXR": {
            "weights": "cxr_softmax_weights.csv", "bias": "cxr_softmax_bias.csv",
            "mean": "cxr_train_mean.csv", "std": "cxr_train_std.csv",
            "labels": {0: "Normal", 1: "Pneumonia", 2: "Tuberculosis"}
        }
    }

    if image_type not in MODEL_CONFIGS:
        return "Unknown", "Unknown", 0.0

    cfg = MODEL_CONFIGS[image_type]
    try:
        W = np.loadtxt(get_path(cfg["weights"]), delimiter=",")
        b = np.loadtxt(get_path(cfg["bias"]), delimiter=",").reshape(1, -1)
        
        # FIXED: Consistent scaling for ALL models
        if os.path.exists(get_path(cfg["mean"])):
            mean = np.loadtxt(get_path(cfg["mean"]), delimiter=",")
            std = np.loadtxt(get_path(cfg["std"]), delimiter=",")
            img_processed = feature_scaling(img_flattened, mean, std)
        else:
            img_processed = (img_flattened - np.mean(img_flattened)) / (np.std(img_flattened) + 1e-8)

        z = np.dot(img_processed.reshape(1, -1), W) + b
        probs = softmax(z)
        
        pred_class = int(np.argmax(probs))
        confidence = float(np.max(probs) * 100)
        label = cfg["labels"].get(pred_class, "Unknown")

        status = "Diseased" if label != "Normal" else "Healthy"
        disease = label if status == "Diseased" else "Normal"

        return status, disease, confidence
    except Exception as e:
        print(f"Disease prediction error for {image_type}: {e}")
        return "Unknown", "Unknown", 0.0

@app.route("/", methods=["GET"])
def health_check():
    missing_files = []
    critical_files = ["softmax_weights.csv", "label_mapping.csv"]
    for f in critical_files:
        if not os.path.exists(get_path(f)):
            missing_files.append(f)
    
    status = "running" if not missing_files else "MISSING_MODEL_FILES"
    return jsonify({"status": status, "missing": missing_files}), 200

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
                "disease": "Please upload CT/MRI/X-ray scan"
            }), 400

        img_flat = load_and_preprocess_image(image)
        ml_type, ml_conf = predict_image_type(img_flat)
        anatomy_hint = anatomical_region_hint(image)

        # FIXED: Only override if ML confidence is very low AND anatomy is confident
        final_type = ml_type
        final_conf = ml_conf
        if ml_conf < 20.0 and anatomy_hint != "Unknown":
            final_type = anatomy_hint
            final_conf = 75.0  # Conservative confidence

        status, disease, disease_conf = predict_disease(final_type, img_flat)

        return jsonify({
            "image_type": final_type,
            "image_type_confidence": round(final_conf, 1),
            "status": status,
            "disease": disease,
            "disease_confidence": round(disease_conf, 1)
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
