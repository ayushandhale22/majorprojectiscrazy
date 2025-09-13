import os
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# === Initialize Flask App ===
app = Flask(__name__)

# === Load your trained model ===
# Make sure you uploaded your model file (model.h5) in the project directory
MODEL_PATH = "model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# === Health Check Route ===
@app.route("/")
def home():
    return "🚀 App is running successfully on Railway!"

# === Prediction Route ===
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        # Preprocess image
        img = image.load_img(file, target_size=(224, 224))  # change size to match your model
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # normalize if your model needs it

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]

        return jsonify({
            "status": "success",
            "predicted_class": int(predicted_class),
            "raw_prediction": prediction.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Main Entry Point ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Railway assigns PORT dynamically
    app.run(host="0.0.0.0", port=port)
