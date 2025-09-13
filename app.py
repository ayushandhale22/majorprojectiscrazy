import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

# === Hugging Face model URL ===
MODEL_URL = "https://huggingface.co/godoffireandiceandknight/effnet_v2_hugface_upload/resolve/main/effnet_v2.keras"

# === Download and load model (cached in ~/.keras/datasets) ===
model_path = tf.keras.utils.get_file("effnet_v2.keras", MODEL_URL)
model = tf.keras.models.load_model(model_path)

# === Class names mapping (use your own trained labels) ===
class_names = [
    "Non-Recyclable & Biodegradable",
    "Non-Recyclable & Non-Biodegradable",
    "Recyclable & Biodegradable",
    "Recyclable & Non-Biodegradable"
]

app = Flask(__name__)

@app.route("/")
def home():
    return {"status": "ok", "message": "API is running 🚀"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get uploaded image
        file = request.files["image"]

        # Preprocess
        img = tf.keras.utils.load_img(file, target_size=(224, 224))
        x = tf.keras.utils.img_to_array(img)
        x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
        x = np.expand_dims(x, axis=0)

        # Predict
        preds = model.predict(x)
        class_idx = int(np.argmax(preds))
        class_label = class_names[class_idx]

        return jsonify({"class_id": class_idx + 1, "class_label": class_label})
    except Exception as e:
        return jsonify({"error": str(e)})
