from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import cv2
import numpy as np
import os

from model import GCPModel

app = Flask(__name__)
CORS(app)

# Device setup
device = torch.device("cpu")

# Load model
model = GCPModel()
model.load_state_dict(
    torch.load("../model_files/best_model.pth", map_location=device)
)
model.eval()

# Shape labels
reverse_shape_map = {
    0: "Cross",
    1: "Square",
    2: "L-Shape"
}

# Root route (for checking server health)
@app.route("/")
def home():
    return "GCP Pose Estimation Backend is Running"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    # Convert uploaded image into OpenCV format
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image file"}), 400

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w, _ = image.shape

    # Center crop strategy
    center_x = w // 2
    center_y = h // 2
    half = 128

    x1 = max(0, center_x - half)
    y1 = max(0, center_y - half)
    x2 = min(w, center_x + half)
    y2 = min(h, center_y + half)

    patch = image[y1:y2, x1:x2]

    # Resize patch to model input size
    patch = cv2.resize(patch, (256, 256))

    # Convert to tensor
    patch = torch.tensor(patch).permute(2, 0, 1).float() / 255.0
    patch = patch.unsqueeze(0)

    # Prediction
    with torch.no_grad():
        pred_coords, pred_shape = model(patch)

    # Convert relative coords back to absolute image coords
    pred_x = x1 + (pred_coords[0][0].item() * 256)
    pred_y = y1 + (pred_coords[0][1].item() * 256)

    # Shape classification
    shape_idx = torch.argmax(pred_shape, dim=1).item()

    return jsonify({
        "x": float(pred_x),
        "y": float(pred_y),
        "shape": reverse_shape_map[shape_idx]
    })

# Render deployment
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )