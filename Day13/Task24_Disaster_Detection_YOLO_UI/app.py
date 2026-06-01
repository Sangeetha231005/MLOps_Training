from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import cv2
import base64
import numpy as np

app = Flask(__name__)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    return response

# Load model
model = YOLO("best.pt")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    try:
        # Read file bytes directly into numpy array
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid or corrupted image format"}), 400

        # Predict using YOLO directly on the numpy array in memory
        results = model.predict(
            source=img,
            conf=0.25,
            save=False
        )

        output = []
        output_image = None

        for result in results:
            # Plot predictions on the frame
            annotated_frame = result.plot()
            
            # Encode to JPEG
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            
            # Convert to Base64
            output_image = base64.b64encode(buffer).decode('utf-8')

            boxes = result.boxes

            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Bounding box coordinates: [x1, y1, x2, y2]
                    xyxy = box.xyxy[0].tolist()

                    output.append({
                        "class_id": cls,
                        "class_name": model.names.get(cls, f"Class {cls}"),
                        "confidence": round(conf, 4),
                        "bbox": [round(coord, 2) for coord in xyxy]
                    })

        return jsonify({
            "detections": output,
            "image": output_image
        })

    except Exception as e:
        return jsonify({"error": f"Internal prediction error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
