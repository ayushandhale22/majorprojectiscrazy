import os
import requests
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image

# === Flask App ===
app = Flask(__name__)

# === Hugging Face Model URL ===
MODEL_URL = "https://huggingface.co/godoffireandiceandknight/effnet_v2_hugface_upload/resolve/main/effnet_v2.keras"
MODEL_PATH = "effnet_v2.keras"

# === Download Model if not present ===
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Hugging Face...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("✅ Model downloaded!")

# === Load Model ===
print("⚡ Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# === Health Check Route ===
@app.route("/")
def home():
    return "🚀 Model API is running on Railway with Hugging Face model!"

# === Prediction Route ===
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        # Preprocess image (adjust size if your model expects different input)
        img = image.load_img(file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # normalize

        # Prediction
        preds = model.predict(img_array)
        predicted_class = np.argmax(preds, axis=1)[0]

        return jsonify({
            "status": "success",
            "predicted_class": int(predicted_class),
            "raw_prediction": preds.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Run App ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Railway sets PORT dynamically
    app.run(host="0.0.0.0", port=port)
