"""
Flask backend for the MNIST Digit Classifier Web UI.
Loads the trained PyTorch model and serves predictions via a REST API.
"""

import io
import base64
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
from torchvision.transforms import Compose, Grayscale, Resize, ToTensor
from torchnn import ImageClassifier, get_device

app = Flask(__name__)

# ── Load model once at startup ───────────────────────────────────────────

device = get_device()
model = ImageClassifier().to(device)

MODEL_PATH = "model_state.pt"

try:
    with open(MODEL_PATH, "rb") as f:
        model.load_state_dict(
            torch.load(f, map_location=device, weights_only=True)
        )

    model.eval()

    # Store activations captured from CNN layers
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu()
        return hook

    # Attach hooks to Conv2d and ReLU layers
    for name, layer in model.model.named_modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.ReLU)):
            layer.register_forward_hook(hook_fn(name))

    print(f"[ok] Model loaded from {MODEL_PATH} on {device}")

except FileNotFoundError:
    print(f"[ERROR] Model file not found: {MODEL_PATH}")
    print("    Run: python torchnn.py --epochs 10")
    exit(1)


# ── Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Accept a base64-encoded image and return digit probabilities."""

    data = request.get_json()

    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    # Decode base64 image
    image_data = data["image"].split(",")[1]
    image_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(image_bytes))

    # Preprocess: convert to grayscale and resize to 28x28
    transform = Compose([
        Grayscale(num_output_channels=1),
        Resize((28, 28)),
        ToTensor(),
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)

    # Check if the canvas contains any significant pixel
    max_val = img_tensor.max().item()

    if max_val < 0.1:
        return jsonify({
            "prediction": -1,
            "confidence": 0.0,
            "probabilities": [0.0] * 10,
            "message": "Canvas appears empty"
        })

    # Clear activations from the previous prediction
    activations.clear()

    # Run the CNN
    with torch.no_grad():
        logits = model(img_tensor)

    # Print captured activations for debugging
    print("\n--- ACTIVATIONS ---")

    for name, activation in activations.items():
        print(name, tuple(activation.shape))

    # Convert model output into probabilities
    probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred = int(np.argmax(probs))
    confidence = float(probs[pred])

    return jsonify({
        "prediction": pred,
        "confidence": confidence,
        "probabilities": [float(p) for p in probs],
    })


# ── Start Flask server ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n   Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5000)