from pyexpat import features

from flask import Flask, request, jsonify, send_from_directory
import os
import joblib
from tensorflow.keras.models import load_model
import numpy as np

from utils import FeatureExtractor


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
MODELS_DIR = os.path.join(BASE_DIR, "models")


# --------------------------------------------------
# Load trained model artifacts
# --------------------------------------------------

MODEL_PATH = os.path.join(MODELS_DIR, "mlp_model.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")


print("Loading trained model...")

model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

feature_extractor = FeatureExtractor()

print("Model loaded successfully.")
print(f"Model: {type(model).__name__}")
print(f"Classes: {len(label_encoder.classes_)}")


# --------------------------------------------------
# Flask application
# --------------------------------------------------

app = Flask(
    __name__,
    static_folder=WEB_DIR,
    static_url_path=""
)


# --------------------------------------------------
# Frontend
# --------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


# --------------------------------------------------
# Prediction API
# --------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if not data or "landmarks" not in data:
            return jsonify({
                "success": False,
                "error": "No landmarks received."
            }), 400

        landmarks = data["landmarks"]

        # Must contain exactly 21 MediaPipe landmarks
        if len(landmarks) != 21:
            return jsonify({
                "success": False,
                "error": f"Expected 21 landmarks, received {len(landmarks)}."
            }), 400

        # Convert landmarks to numpy array
        landmarks = np.array(landmarks, dtype=np.float64)

        # Extract the SAME 93 features used during training
        features = feature_extractor.extract(landmarks)

        if features is None:
            return jsonify({
                "success": False,
                "error": "Feature extraction failed."
            }), 400
        
        # Scale features using the trained scaler
        features_scaled = scaler.transform([features])

        # Keras MLP prediction
        probabilities = model(features_scaled, training=False).numpy()[0]

        # Get predicted class index
        prediction = int(np.argmax(probabilities))

        # Convert encoded class number to actual label
        label = label_encoder.inverse_transform([prediction])[0]

        # Confidence = highest class probability
        confidence = float(np.max(probabilities))

        return jsonify({
            "success": True,
            "label": str(label),
            "confidence": confidence
        })

    except Exception as e:

        print("Prediction error:", e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# --------------------------------------------------
# Run server
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print(" ASL SIGN LANGUAGE RECOGNITION")
    print("=" * 60)
    print()
    print("Frontend + ML Backend")
    print()
    print("Open:")
    print("http://localhost:5000")
    print()
    print("Press CTRL+C to stop.")
    print("=" * 60)

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )