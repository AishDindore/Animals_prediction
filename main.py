from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# ================= LOAD MODEL =================
model = tf.keras.models.load_model("Animals_Images_Prediction.keras")

class_names = [
    "Bear","Bird","Cat","Cow","Deer","Dog",
    "Dolphine","Elephant","Giraffe","Horse",
    "Kangaroo","Lion","Pandas","Tiger","Zebra"
]

IMAGE_SIZE = (224, 224)

# ================= UI ROUTE =================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
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


# ================= API ROUTE (FOR POSTMAN) =================
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
    app.run(host="0.0.0.0", port=5000)
