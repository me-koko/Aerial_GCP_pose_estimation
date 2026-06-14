from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import cv2
import numpy as np
from model import GCPModel

app = Flask(__name__)
CORS(app)

device = torch.device("cpu")

model = GCPModel()
model.load_state_dict(torch.load("../model_files/best_model.pth", map_location=device))
model.eval()

reverse_shape_map = {
    0: "Cross",
    1: "Square",
    2: "L-Shape"
}

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w, _ = image.shape

    center_x = w // 2
    center_y = h // 2
    half = 128

    x1 = max(0, center_x - half)
    y1 = max(0, center_y - half)
    x2 = min(w, center_x + half)
    y2 = min(h, center_y + half)

    patch = image[y1:y2, x1:x2]

    patch = cv2.resize(patch, (256, 256))
    patch = torch.tensor(patch).permute(2, 0, 1).float() / 255.0
    patch = patch.unsqueeze(0)

    with torch.no_grad():
        pred_coords, pred_shape = model(patch)

    pred_x = x1 + (pred_coords[0][0].item() * 256)
    pred_y = y1 + (pred_coords[0][1].item() * 256)

    shape_idx = torch.argmax(pred_shape, dim=1).item()

    return jsonify({
        "x": float(pred_x),
        "y": float(pred_y),
        "shape": reverse_shape_map[shape_idx]
    })

if __name__ == "__main__":
    app.run(debug=True)