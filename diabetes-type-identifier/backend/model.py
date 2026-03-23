"""
model.py - ML model definition, training, and inference
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import os

from preprocessing import (
    FEATURE_COLUMNS, LABEL_MAP, preprocess_dataframe, load_dataset, save_artifacts
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def train_model(df: pd.DataFrame, model_type: str = "random_forest") -> dict:
    """
    Train a classifier on the provided dataframe.
    Expects an 'Outcome' column: 0=No Diabetes, 1=Type1, 2=Type2
    Returns metrics and saves model artifacts.
    """
    X_scaled, scaler, imputer = preprocess_dataframe(df, fit=True)
    y = df["Outcome"].values

    # Handle class imbalance with SMOTE
    try:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    except Exception:
        # SMOTE needs at least 2 samples per class; fall back if not enough data
        X_resampled, y_resampled = X_scaled, y

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    if model_type == "decision_tree":
        clf = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42)
    else:
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            random_state=42, n_jobs=-1
        )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics = {
        "f1_score": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)],
            output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "model_type": model_type,
        "training_samples": len(X_resampled),
    }

    # Save artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(MODELS_DIR, "trained_model.pkl"))
    save_artifacts(scaler, imputer, MODELS_DIR)

    return metrics


def predict(patient_data: dict) -> dict:
    """
    Run inference on a single patient record.
    Returns predicted label, confidence, and feature importances.
    """
    clf, scaler, imputer = load_model_artifacts()

    df = pd.DataFrame([patient_data])
    X_scaled, _, _ = preprocess_dataframe(df, scaler=scaler, imputer=imputer, fit=False)

    proba = clf.predict_proba(X_scaled)[0]
    pred_class = int(np.argmax(proba))
    classes = clf.classes_

    # Map probabilities to label names
    confidence_scores = {
        LABEL_MAP[int(c)]: round(float(p), 4)
        for c, p in zip(classes, proba)
    }

    # Feature importance
    feature_importance = {}
    if hasattr(clf, "feature_importances_"):
        available_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        for feat, imp in zip(available_cols, clf.feature_importances_):
            feature_importance[feat] = round(float(imp), 4)

    return {
        "predicted_class": LABEL_MAP[pred_class],
        "confidence": round(float(proba[list(classes).index(pred_class)]), 4),
        "confidence_scores": confidence_scores,
        "feature_importance": dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        ),
    }


def predict_batch(df: pd.DataFrame) -> list:
    """Run inference on a dataframe of patients."""
    clf, scaler, imputer = load_model_artifacts()
    X_scaled, _, _ = preprocess_dataframe(df, scaler=scaler, imputer=imputer, fit=False)

    probas = clf.predict_proba(X_scaled)
    classes = clf.classes_
    results = []

    for i, proba in enumerate(probas):
        pred_class = int(np.argmax(proba))
        results.append({
            "row": i + 1,
            "predicted_class": LABEL_MAP[pred_class],
            "confidence": round(float(proba[list(classes).index(pred_class)]), 4),
            "confidence_scores": {
                LABEL_MAP[int(c)]: round(float(p), 4)
                for c, p in zip(classes, proba)
            },
        })

    return results


def load_model_artifacts():
    """Load trained model, scaler, and imputer from disk."""
    model_path = os.path.join(MODELS_DIR, "trained_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    imputer_path = os.path.join(MODELS_DIR, "imputer.pkl")

    if not all(os.path.exists(p) for p in [model_path, scaler_path, imputer_path]):
        raise FileNotFoundError(
            "Model artifacts not found. Please train the model first via POST /train"
        )

    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    imputer = joblib.load(imputer_path)
    return clf, scaler, imputer


def model_is_trained() -> bool:
    model_path = os.path.join(MODELS_DIR, "trained_model.pkl")
    return os.path.exists(model_path)
