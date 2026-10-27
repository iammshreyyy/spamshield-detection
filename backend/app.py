import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": os.getenv("FRONTEND_ORIGIN", "*")}})

MODEL_PATH = os.path.join(os.path.dirname(__file__), "spam_model.joblib")

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.post("/api/predict")
def predict():
    global model
    if model is None:
        return jsonify({
            "error": "Model not trained yet. Add a labeled CSV and run train_model.py."
        }), 503

    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    channel = str(data.get("channel", "sms")).lower()

    if not message:
        return jsonify({"error": "Please provide a message."}), 400
    if channel not in ("sms", "email"):
        return jsonify({"error": "channel must be 'sms' or 'email'."}), 400

    # For email, send subject + body together from the frontend.
    prediction = str(model.predict([message])[0]).lower()
    scores = model.predict_proba([message])[0]
    classes = list(model.classes_)
    confidence = float(scores[classes.index(prediction)])

    return jsonify({
        "channel": channel,
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "warning": "Model predictions can be wrong; do not automatically delete or block messages."
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
