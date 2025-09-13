import os
import numpy as np
import requests
from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model

app = Flask(__name__)

# ==============================
# Build Model from Scratch
# ==============================
print("Building model...")

num_classes = 10  # ⚡️ change to your dataset’s number of classes

inputs = Input(shape=(224, 224, 3))
base = EfficientNetV2B0(include_top=False, weights="imagenet")(inputs)  # pretrained backbone
x = Flatten()(base)
x = Dense(256, activation="relu")(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs)

print("✅ Model built successfully!")

# ==============================
# Prediction Route
# ==============================
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        # Preprocess image
        img = Image.open(file).convert("RGB").resize((224, 224))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        # Prediction
        preds = model.predict(arr)
        preds_list = preds.tolist()

        return jsonify({"prediction": preds_list})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==============================
# Health Check
# ==============================
@app.route("/", methods=["GET"])
def home():
    return "✅ Model API is running!"

# ==============================
# Run App
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
