from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# ================= GLOBAL VARIABLES =================
model = None
IMAGE_SIZE = (224, 224)

class_names = [
    "Bear", "Bird", "Cat", "Cow", "Deer", "Dog",
    "Dolphine", "Elephant", "Giraffe", "Horse",
    "Kangaroo", "Lion", "Pandas", "Tiger", "Zebra"
]

# ================= LOAD MODEL LAZILY =================
def load_model_once():
    global model
    if model is None:
        model = tf.keras.models.load_model("Animals_Images_Prediction.keras")

# ================= UI ROUTE =================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        load_model_once()

        file = request.files.get("image")
        if file:
            os.makedirs("static", exist_ok=True)
            image_path = os.path.join("static", file.filename)
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
        load_model_once()

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
