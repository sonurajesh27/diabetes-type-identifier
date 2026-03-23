"""
app.py - Flask REST API for AI Diabetes Type Identifier
"""

import os
import io
import json
import traceback

import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Ensure backend/ is on path for sibling imports
import sys
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import load_dataset, FEATURE_COLUMNS, LABEL_MAP
from model import train_model, predict, predict_batch, model_is_trained

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


# ─── Root ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "app": "AI Diabetes Type Identifier",
        "version": "1.0.0",
        "endpoints": ["/health", "/train", "/predict", "/features"]
    })


# ─── Health Check ────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_trained": model_is_trained(),
        "version": "1.0.0"
    })


# ─── Train ───────────────────────────────────────────────────────────────────

@app.route("/train", methods=["POST"])
def train():
    """
    Train the model.
    Accepts: multipart/form-data with 'file' (CSV) OR uses default sample data.
    Optional form field: model_type = random_forest | decision_tree
    """
    model_type = request.form.get("model_type", "random_forest")

    try:
        if "file" in request.files:
            file = request.files["file"]
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            df = load_dataset(filepath)
        else:
            default_path = os.path.join(
                os.path.dirname(__file__), "..", "data", "sample_diabetes.csv"
            )
            df = load_dataset(default_path)

        metrics = train_model(df, model_type=model_type)
        return jsonify({"success": True, "metrics": metrics})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ─── Predict ─────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    Predict diabetes type.

    Accepts:
      - application/json: single patient dict
      - multipart/form-data with 'file': CSV for batch prediction
    """
    if not model_is_trained():
        return jsonify({
            "success": False,
            "error": "Model not trained yet. Call POST /train first."
        }), 400

    try:
        # ── Batch CSV prediction ──
        if "file" in request.files:
            file = request.files["file"]
            df = pd.read_csv(io.StringIO(file.read().decode("utf-8")))
            results = predict_batch(df)
            return jsonify({"success": True, "predictions": results, "count": len(results)})

        # ── Single patient JSON prediction ──
        data = request.get_json(force=True)
        if not data:
            return jsonify({"success": False, "error": "No input data provided"}), 400

        result = predict(data)
        return jsonify({"success": True, **result})

    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ─── Feature Info ────────────────────────────────────────────────────────────

@app.route("/features", methods=["GET"])
def features():
    """Return expected feature names and label mapping."""
    return jsonify({
        "features": FEATURE_COLUMNS,
        "labels": LABEL_MAP
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
