from flask import Flask, request, jsonify
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load model
model = YOLO("best.pt")

@app.route("/")
def home():
    return "Victim Detection YOLO Model Running"

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["image"]

    image_path = "temp.jpg"

    file.save(image_path)

    results = model.predict(
        source=image_path,
        conf=0.25,
        save=False
    )

    output = []

    for result in results:

        boxes = result.boxes

        if boxes is not None:

            for box in boxes:

                cls = int(box.cls[0])

                conf = float(box.conf[0])

                output.append({
                    "class_id": cls,
                    "confidence": round(conf, 4)
                })

    os.remove(image_path)

    return jsonify(output)

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
