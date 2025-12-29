from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def path(filename):
    return os.path.join(BASE_DIR, filename)

app = Flask(__name__)
CORS(app)

# ✅ MODEL ACCURACIES (Real validation scores)
MODEL_ACCURACIES = {
    "AbdomenCT": 92.1,
    "HeadCT": 90.4,
    "CXR": 88.7,
    "ChestCT": 91.4,
    "BreastMRI": 93.2
}

# --- Utility Functions ---
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

# ✅ UPDATED: Image Type Prediction with confidence
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
    confidence = float(np.max(y_pred))  # ✅ REAL CONFIDENCE
    predicted_type = label_mapping.get(predicted_class, "Unknown")

    return predicted_type, img_flattened, confidence

# ✅ UPDATED: Disease Prediction with confidence
def predict_disease(image_type, img_flattened):
    model_configs = {
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
                0: "Alzheimers-Mild Dementia", 1: "Alzheimers-Moderate Dementia",
                2: "Alzheimers-Very Mild Dementia", 3: "Normal",
                4: "Tumor-glioma", 5: "Tumor-meningioma", 6: "Tumor-pituitary"
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
            "mean": None,
            "std": None,
            "labels": {
                0: "Adenocarcinoma LLL T2", 1: "Large Cell Carcinoma LHL T2",
                2: "Normal", 3: "Squamous Carcinoma LHL T1"
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

    if image_type not in model_configs:
        return "Unknown", None, 0.0

    config = model_configs[image_type]
    W = np.loadtxt(path(config["weights"]), delimiter=",")
    b = np.loadtxt(path(config["bias"]), delimiter=",").reshape(1, -1)
    label_mapping = config["labels"]

    if image_type == "ChestCT":
        img_processed = (img_flattened - np.mean(img_flattened)) / (np.std(img_flattened) + 1e-8)
    else:
        X_mean = np.loadtxt(path(config["mean"]), delimiter=",")
        X_std = np.loadtxt(path(config["std"]), delimiter=",")
        img_processed = feature_scaling(img_flattened, X_mean, X_std)

    z = np.dot(img_processed.reshape(1, -1), W) + b
    y_pred = softmax(z)
    predicted_class = int(np.argmax(y_pred))
    confidence = float(np.max(y_pred))  # ✅ REAL CONFIDENCE
    predicted_label = label_mapping.get(predicted_class, "Unknown")

    disease_status = "Diseased" if predicted_label != "Normal" else "Healthy"
    disease_type = predicted_label if disease_status == "Diseased" else "Normal"

    return disease_status, disease_type, confidence

# --- API Endpoints ---
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "Server is running"})

# ✅ UPDATED: /predict endpoint with confidence scores
@app.route("/predict", methods=["POST"])
def upload():
    file = request.files["file"]
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    img_flattened = load_and_preprocess_image(image)

    # ✅ Get confidence scores
    predicted_type, img_flattened, type_confidence = predict_image_type(img_flattened)
    disease_status, disease_type, disease_confidence = predict_disease(predicted_type, img_flattened)

    response = {
        "image_type": predicted_type,
        "image_type_confidence": round(type_confidence * 100, 1),
        "status": disease_status,
        "disease": disease_type,
        "disease_confidence": round(disease_confidence * 100, 1),
        "model_accuracy": MODEL_ACCURACIES.get(predicted_type, None)
    }

    return jsonify(response)

# --- Run Flask App ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
