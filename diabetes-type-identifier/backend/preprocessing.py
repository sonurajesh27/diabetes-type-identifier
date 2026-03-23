"""
preprocessing.py - Data preprocessing utilities for diabetes classification
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

FEATURE_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# Columns where 0 is biologically invalid and should be treated as missing
ZERO_AS_NAN_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

LABEL_MAP = {0: "No Diabetes", 1: "Type 1 Diabetes", 2: "Type 2 Diabetes"}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def replace_zeros_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace biologically invalid zero values with NaN."""
    df = df.copy()
    for col in ZERO_AS_NAN_COLS:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    return df


def preprocess_dataframe(df: pd.DataFrame, scaler=None, imputer=None, fit=False):
    """
    Full preprocessing pipeline:
    - Replace invalid zeros
    - Impute missing values
    - Scale features

    Returns: (X_processed, scaler, imputer)
    """
    df = replace_zeros_with_nan(df)

    # Keep only known feature columns that exist
    available_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[available_cols].copy()

    # Impute missing values with median
    if fit or imputer is None:
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)
    else:
        X_imputed = imputer.transform(X)

    # Scale features
    if fit or scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
    else:
        X_scaled = scaler.transform(X_imputed)

    return X_scaled, scaler, imputer


def preprocess_single_patient(patient_dict: dict, scaler, imputer) -> np.ndarray:
    """Preprocess a single patient record from a dict."""
    df = pd.DataFrame([patient_dict])
    X_scaled, _, _ = preprocess_dataframe(df, scaler=scaler, imputer=imputer, fit=False)
    return X_scaled


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load and validate a CSV dataset."""
    df = pd.read_csv(filepath)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df


def save_artifacts(scaler, imputer, models_dir: str = "../models"):
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(imputer, os.path.join(models_dir, "imputer.pkl"))


def load_artifacts(models_dir: str = "../models"):
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    imputer = joblib.load(os.path.join(models_dir, "imputer.pkl"))
    return scaler, imputer
