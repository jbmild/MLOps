"""Feature engineering and preprocessing for Stroke Prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_STATE = 42
TARGET_COL = "stroke"
TEST_SIZE = 0.3

NUMERIC_FEATURES = [
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "avg_glucose_level",
    "bmi",
]

CATEGORICAL_FEATURES = [
    "gender",
    "work_type",
    "Residence_type",
    "smoking_status",
    "bmi_category",
    "glucose_category",
    "age_group",
]

RAW_CATEGORICAL_COLUMNS = [
    "gender",
    "work_type",
    "Residence_type",
    "smoking_status",
]


def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning steps from the AMq1 notebook."""
    df = df.copy()
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    df = df[df["gender"] != "Other"].reset_index(drop=True)
    df["ever_married"] = (df["ever_married"] == "Yes").astype(int)
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    return df


def add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create clinical category features from continuous variables."""
    df = df.copy()
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 24.9, 29.9, np.inf],
        labels=["bajo_peso", "normal", "sobrepeso", "obeso"],
    )
    df["glucose_category"] = pd.cut(
        df["avg_glucose_level"],
        bins=[0, 70, 100, 125, np.inf],
        labels=["bajo", "normal", "prediabetes", "diabetes"],
    )
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 18, 35, 50, 65, np.inf],
        labels=["<18", "18-35", "36-50", "51-65", ">65"],
    )
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns and cast dummies to int."""
    df = df.copy()
    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)
    numeric_cols = NUMERIC_FEATURES + [TARGET_COL]
    dummy_cols = [col for col in df.columns if col not in numeric_cols]
    if dummy_cols:
        df[dummy_cols] = df[dummy_cols].astype(int)
    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline: clean → feature engineering → encoding."""
    df = clean_raw_dataframe(df)
    df = add_clinical_features(df)
    return encode_features(df)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model feature column names (all columns except target)."""
    return [col for col in df.columns if col != TARGET_COL]


def build_data_metadata(df_before_dummy: pd.DataFrame, df_encoded: pd.DataFrame) -> dict:
    """Build metadata dict stored in MinIO as data.json."""
    feature_columns = get_feature_columns(df_encoded)
    dummy_columns = [col for col in feature_columns if col not in NUMERIC_FEATURES]

    category_values = {}
    for col in CATEGORICAL_FEATURES:
        if col in df_before_dummy.columns:
            category_values[col] = sorted(df_before_dummy[col].astype(str).unique().tolist())

    return {
        "target_col": TARGET_COL,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "raw_categorical_columns": RAW_CATEGORICAL_COLUMNS,
        "feature_columns": feature_columns,
        "dummy_columns": dummy_columns,
        "categories_values_per_categorical": category_values,
        "optimal_threshold": 0.441,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }


def get_preprocessor(feature_columns: list[str]):
    """Return sklearn ColumnTransformer matching the AMq1 notebook."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    dummy_features = [col for col in feature_columns if col not in NUMERIC_FEATURES]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                NUMERIC_FEATURES,
            ),
            ("cat", "passthrough", dummy_features),
        ],
    )
