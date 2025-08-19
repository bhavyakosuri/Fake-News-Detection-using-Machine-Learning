import os
import json
from flask import Flask, request, render_template, jsonify
from joblib import load

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")

app = Flask(__name__)

# Lazy loaded model
_model = None
def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run train.py first.")
        _model = load(MODEL_PATH)
    return _model

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    proba = None
    text = ""
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if text:
            model = get_model()
            pred = model.predict([text])[0]
            label = "REAL" if int(pred) == 1 else "FAKE"
            # Some linear models expose decision_function; we map to a pseudo-probability with a sigmoid-ish transform
            score = None
            if hasattr(model.named_steps["clf"], "decision_function"):
                import numpy as np
                s = model.named_steps["clf"].decision_function(model.named_steps["tfidf"].transform([text]))[0]
                # scaled confidence in [0,1]
                score = float(1 / (1 + np.exp(-s)))
                proba = score if pred == 1 else 1 - score
            prediction = label
    return render_template("index.html", prediction=prediction, proba=proba, text=text)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Provide non-empty 'text'"}), 400
    model = get_model()
    pred = model.predict([text])[0]
    label = "REAL" if int(pred) == 1 else "FAKE"
    result = {"prediction": label, "label_id": int(pred)}
    return jsonify(result), 200

if __name__ == "__main__":
    app.run(debug=True)
