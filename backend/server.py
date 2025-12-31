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
    try:
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
    except Exception as e:
        print(f"Type prediction error: {e}")
        return "Unknown", img_flattened

def predict_disease(image_type, img_flattened):
    model_configs = {
        "AbdomenCT": {
            "weights": "abdomen_softmax_weights.csv",
            "bias": "abdomen_softmax_bias.csv",
            "labels": {0: "Cyst", 1: "Normal", 2: "Stone", 3: "Tumor"}
        },
        "HeadCT": {
            "weights": "head_ct_softmax_weights.csv",
            "bias": "head_ct_softmax_bias.csv",
            "labels": {
                0: "Alzheimers-Mild Dementia", 1: "Alzheimers-Moderate Dementia",
                2: "Alzheimers-Very Mild Dementia", 3: "Normal",
                4: "Tumor-glioma", 5: "Tumor-meningioma", 6: "Tumor-pituitary"
            }
        },
        "CXR": {
            "weights": "cxr_softmax_weights.csv",
            "bias": "cxr_softmax_bias.csv",
            "labels": {0: "Normal", 1: "Pneumonia", 2: "Tuberculosis"}
        },
        "ChestCT": {
            "weights": "chest_softmax_weights.csv",
            "bias": "chest_softmax_bias.csv",
            "labels": {
                0: "Adenocarcinoma LLL T2", 1: "Large Cell Carcinoma LHL T2",
                2: "Normal", 3: "Squamous Carcinoma LHL T1"
            }
        },
        "BreastMRI": {
            "weights": "breast_softmax_weights.csv",
            "bias": "breast_softmax_bias.csv",
            "labels": {0: "Cancer", 1: "Normal"}
        }
    }

    if image_type not in model_configs:
        return "Unknown", None

    config = model_configs[image_type]
    try:
        W = np.loadtxt(get_path(config["weights"]), delimiter=",")
        b = np.loadtxt(get_path(config["bias"]), delimiter=",").reshape(1, -1)
        label_mapping = config["labels"]

        # 🔥 MAGIC FIX: SELF-NORMALIZATION FOR ALL MODELS (works for your data!)
        img_processed = (img_flattened - np.mean(img_flattened)) / (np.std(img_flattened) + 1e-8)

        z = np.dot(img_processed.reshape(1, -1), W) + b
        y_pred = softmax(z)
        predicted_class = int(np.argmax(y_pred))
        predicted_label = label_mapping.get(predicted_class, "Unknown")

        disease_status = "Diseased" if predicted_label != "Normal" else "Healthy"
        disease_type = predicted_label if disease_status == "Diseased" else "Normal"

        return disease_status, disease_type
    except Exception as e:
        print(f"Disease prediction failed for {image_type}: {e}")
        return "Unknown", "Unknown"

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "running"}), 200

@app.route("/predict", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image"}), 400

        img_flattened = load_and_preprocess_image(image)
        predicted_type, img_flattened = predict_image_type(img_flattened)
        disease_status, disease_type = predict_disease(predicted_type, img_flattened)

        response = {
            "image_type": predicted_type,
            "status": disease_status,
            "disease": disease_type
        }

        return jsonify(response)

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
