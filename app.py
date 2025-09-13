import os
import requests
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.models import Model

app = Flask(__name__)

# ==============================
# Download Model from Hugging Face
# ==============================
MODEL_URL = "https://huggingface.co/godoffireandiceandknight/effnet_v2_hugface_upload/resolve/main/effnet_v2.keras"
MODEL_PATH = "effnet_v2.keras"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Hugging Face...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("Model downloaded!")

# ==============================
# Load Model and Fix Dense Input
# ==============================
print("Loading model...")
base_model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Wrap base_model output with Flatten and new Dense head if needed
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)  # Output might be 4D
x = Flatten()(x)  # Flatten 4D tensor into 2D
x = Dense(256, activation="relu")(x)
num_classes = x.shape[-1] if hasattr(x, 'shape') else 10  # Adjust num_classes
outputs = Dense(num_classes, activation="softmax")(x)
model = Model(inputs, outputs)

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

