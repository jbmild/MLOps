"""Training utilities: GridSearchCV, Youden threshold, MLflow registration."""

from __future__ import annotations

import mlflow
import numpy as np
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from features import RANDOM_STATE, get_preprocessor


LR_PARAM_GRID = {
    "clf__C": [0.001, 0.01, 0.1, 1, 10],
    "clf__penalty": ["l1", "l2"],
    "clf__solver": ["liblinear"],
}

MODEL_NAME = "stroke_prediction_model_prod"
EXPERIMENT_NAME = "Stroke Prediction"


def build_lr_pipeline(feature_columns: list[str]) -> Pipeline:
    """Build the full training pipeline (preprocessor + LogisticRegression)."""
    return Pipeline([
        ("prep", get_preprocessor(feature_columns)),
        (
            "clf",
            LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def run_grid_search(X_train, y_train, feature_columns: list[str]) -> GridSearchCV:
    """Run GridSearchCV with F1 scoring and 5-fold stratified CV."""
    pipeline = build_lr_pipeline(feature_columns)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipeline,
        LR_PARAM_GRID,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def compute_youden_threshold(model: Pipeline, X_train, y_train) -> float:
    """Compute optimal probability threshold using Youden's index on CV predictions."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_proba = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
    )[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, y_proba)
    return float(thresholds[np.argmax(tpr - fpr)])


def evaluate_model(model: Pipeline, X_test, y_test, threshold: float) -> dict:
    """Evaluate model with default and tuned thresholds."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_default = model.predict(X_test)
    y_pred_tuned = (y_proba >= threshold).astype(int)

    return {
        "f1_default": f1_score(y_test, y_pred_default),
        "f1_tuned": f1_score(y_test, y_pred_tuned),
        "recall_default": recall_score(y_test, y_pred_default),
        "recall_tuned": recall_score(y_test, y_pred_tuned),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def register_champion_model(
    model: Pipeline,
    X_train,
    metrics: dict,
    threshold: float,
    tracking_uri: str = "http://mlflow:5000",
) -> str:
    """Log model to MLflow and register as champion in the model registry."""
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.set_experiment(EXPERIMENT_NAME).experiment_id

    with mlflow.start_run(
        experiment_id=experiment_id,
        tags={"experiment": "hpo", "dataset": "stroke"},
    ):
        mlflow.log_params(model.get_params())
        mlflow.log_metric("optimal_threshold", threshold)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        signature = infer_signature(X_train, model.predict_proba(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name=MODEL_NAME,
            metadata={"optimal_threshold": threshold},
        )
        model_uri = mlflow.get_artifact_uri("model")

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest = max(int(v.version) for v in versions)
    client.set_registered_model_alias(MODEL_NAME, "champion", str(latest))

    return model_uri
