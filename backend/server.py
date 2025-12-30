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
# Allows all origins - necessary for connecting your frontend to Render
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(BASE_DIR, filename)

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
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

# ============================================================
# ML PREDICTION LOGIC
# ============================================================

def predict_image_type(img_flattened):
    # Load model files using the safe path helper
    W = np.loadtxt(get_path("softmax_weights.csv"), delimiter=",")
    b = np.loadtxt(get_path("softmax_bias.csv"), delimiter=",")
    X_mean = np.loadtxt(get_path("train_mean.csv"), delimiter=",")
    X_std = np.loadtxt(get_path("train_std.csv"), delimiter=",")
    
    label_mapping_df = pd.read_csv(get_path("label_mapping.csv"))
    label_mapping = dict(zip(label_mapping_df["Index"], label_mapping_df["Label"]))

    img_processed = feature_scaling(img_flattened, X_mean, X_std)

    z = np.dot(img_processed, W) + b
    y_pred = softmax(z)
    predicted_class = int(np.argmax(y_pred))
    predicted_type = label_mapping.get(predicted_class, "Unknown")

    return predicted_type, img_flattened

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
            "mean": "chest_train_mean.csv",
            "std": "chest_train_std.csv",
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
        return "Unknown", "Unknown"

    config = model_configs[image_type]
    W = np.loadtxt(get_path(config["weights"]), delimiter=",")
    b = np.loadtxt(get_path(config["bias"]), delimiter=",").reshape(1, -1)
    label_mapping = config["labels"]

    # Special handling for ChestCT scaling logic from your original code
    if image_type == "ChestCT":
        img_processed = (img_flattened - np.mean(img_flattened)) / (np.std(img_flattened) + 1e-8)
    else:
        X_mean = np.loadtxt(get_path(config["mean"]), delimiter=",")
        X_std = np.loadtxt(get_path(config["std"]), delimiter=",")
        img_processed = feature_scaling(img_flattened, X_mean, X_std)

    z = np.dot(img_processed.reshape(1, -1), W) + b
    y_pred = softmax(z)
    predicted_class = int(np.argmax(y_pred))
    predicted_label = label_mapping.get(predicted_class, "Unknown")

    disease_status = "Diseased" if predicted_label != "Normal" else "Healthy"
    disease_type = predicted_label if disease_status == "Diseased" else "Normal"

    return disease_status, disease_type

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
        # Convert file stream to OpenCV image
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Step 1: Preprocess
        img_flattened = load_and_preprocess_image(image)

        # Step 2: Prediction Pipeline
        predicted_type, img_flattened = predict_image_type(img_flattened)
        disease_status, disease_type = predict_disease(predicted_type, img_flattened)

        response = {
            "image_type": predicted_type,
            "status": disease_status,
            "disease": disease_type
        }

        return jsonify(response)

    except Exception as e:
        # Logs the full error to Render's console for debugging
        print(traceback.format_exc())
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# ============================================================
# RENDER DEPLOYMENT SETTINGS
# ============================================================
if __name__ == "__main__":
    # Render provides the port via environment variables
    port = int(os.environ.get("PORT", 5000))
    # Must bind to 0.0.0.0 for Render to expose the service
    app.run(host="0.0.0.0", port=port, debug=False)