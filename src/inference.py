"""Inference helpers: raw clinical input → model-ready features → prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RAW_CATEGORICAL_COLUMNS,
    add_clinical_features,
    clean_raw_dataframe,
    encode_features,
)


def raw_input_to_dataframe(features: dict) -> pd.DataFrame:
    """Convert API payload (raw clinical fields) to a single-row DataFrame."""
    row = {
        "gender": features["gender"],
        "age": features["age"],
        "hypertension": features["hypertension"],
        "heart_disease": features["heart_disease"],
        "ever_married": features["ever_married"],
        "work_type": features["work_type"],
        "Residence_type": features["Residence_type"],
        "avg_glucose_level": features["avg_glucose_level"],
        "bmi": features.get("bmi"),
        "smoking_status": features["smoking_status"],
    }
    df = pd.DataFrame([row])
    if isinstance(row["ever_married"], str):
        df["ever_married"] = (df["ever_married"] == "Yes").astype(int)
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    return df


def transform_raw_features(features: dict, data_dict: dict) -> pd.DataFrame:
    """
    Transform raw clinical input into feature matrix aligned with training columns.

    Uses metadata from data.json to ensure dummy column order matches the model.
    """
    df = raw_input_to_dataframe(features)
    df = add_clinical_features(df)

    for col in RAW_CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)

    feature_columns = data_dict["feature_columns"]
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]
    dummy_cols = [c for c in feature_columns if c not in NUMERIC_FEATURES]
    if dummy_cols:
        df[dummy_cols] = df[dummy_cols].astype(int)

    return df


def predict_with_threshold(model, features: dict, data_dict: dict) -> dict:
    """Run inference applying the Youden threshold from training."""
    X = transform_raw_features(features, data_dict)
    threshold = data_dict.get("optimal_threshold", 0.441)
    probability = float(model.predict_proba(X)[0, 1])
    stroke_detected = probability >= threshold
    risk_level = "Alto" if stroke_detected else "Bajo"

    return {
        "stroke_detected": stroke_detected,
        "probability": round(probability, 4),
        "risk_level": risk_level,
        "threshold_used": threshold,
    }
