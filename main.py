from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import uuid
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

print("FILES:", os.listdir("."))

# ================= GLOBAL VARIABLES =================
model = None
IMAGE_SIZE = (224, 224)

class_names = [
    "Bear", "Bird", "Cat", "Cow", "Deer", "Dog",
    "Dolphine", "Elephant", "Giraffe", "Horse",
    "Kangaroo", "Lion", "Pandas", "Tiger", "Zebra"
]

# ================= LOAD MODEL ONCE =================
def load_model_once():
    global model
    if model is None:
        print("Loading model...")
        model_path = os.path.join(os.getcwd(), "Animals_Images_Prediction.h5")
        model = tf.keras.models.load_model(model_path, compile=False)

load_model_once()

# ================= UI ROUTE =================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            os.makedirs("static", exist_ok=True)

            filename = f"{uuid.uuid4()}.jpg"
            image_path = os.path.join("static", filename)
            file.save(image_path)

            img = Image.open(image_path).convert("RGB")
            img = img.resize(IMAGE_SIZE)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)
            prediction = class_names[np.argmax(preds)]
            confidence = round(float(np.max(preds)) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )

# ================= API ROUTE =================
@app.route("/predict-api", methods=["POST"])
def predict_api():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]

        img = Image.open(file).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        predicted_class = class_names[np.argmax(preds)]
        confidence = round(float(np.max(preds)) * 100, 2)

        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True, port=5000)