import json
import pickle

import boto3
import mlflow
from fastapi import BackgroundTasks, Body, FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from inference import predict_with_threshold

MODEL_NAME = "stroke_prediction_model_prod"
ALIAS = "champion"


def load_model(model_name: str, alias: str):
    """Load champion model from MLflow Registry and data.json from MinIO."""
    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        client = mlflow.MlflowClient()
        model_data = client.get_model_version_by_alias(model_name, alias)
        model = mlflow.sklearn.load_model(model_data.source)
        version = int(model_data.version)
    except Exception:
        with open("/app/files/model.pkl", "rb") as f:
            model = pickle.load(f)
        version = 0

    try:
        s3 = boto3.client("s3")
        result = s3.get_object(Bucket="data", Key="data_info/data.json")
        data_dictionary = json.loads(result["Body"].read().decode())
    except Exception:
        with open("/app/files/data.json", "r") as f:
            data_dictionary = json.load(f)

    return model, version, data_dictionary


def check_model():
    """Reload model if champion version changed in MLflow Registry."""
    global model, data_dict, version_model
    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        client = mlflow.MlflowClient()
        new_model_data = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        new_version = int(new_model_data.version)
        if new_version != version_model:
            model, version_model, data_dict = load_model(MODEL_NAME, ALIAS)
    except Exception:
        pass


class ModelInput(BaseModel):
    """Raw clinical features for stroke prediction."""

    gender: str = Field(description='Patient gender: "Male" or "Female"')
    age: float = Field(description="Age in years", ge=0, le=120)
    hypertension: int = Field(description="Hypertension: 0 = no, 1 = yes", ge=0, le=1)
    heart_disease: int = Field(description="Heart disease: 0 = no, 1 = yes", ge=0, le=1)
    ever_married: int = Field(description="Ever married: 0 = no, 1 = yes", ge=0, le=1)
    work_type: str = Field(description='Work type: "Private", "Self-employed", "Govt_job", "children", "Never_worked"')
    Residence_type: str = Field(description='Residence: "Urban" or "Rural"')
    avg_glucose_level: float = Field(description="Average glucose level mg/dL", ge=0)
    bmi: float | None = Field(default=None, description="Body mass index (optional, imputed if missing)")
    smoking_status: str = Field(
        description='Smoking status: "never smoked", "formerly smoked", "smokes", "Unknown"'
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "gender": "Male",
                    "age": 67,
                    "hypertension": 0,
                    "heart_disease": 1,
                    "ever_married": 1,
                    "work_type": "Private",
                    "Residence_type": "Urban",
                    "avg_glucose_level": 228.69,
                    "bmi": 36.6,
                    "smoking_status": "formerly smoked",
                }
            ]
        }
    }


class ModelOutput(BaseModel):
    """Stroke prediction response."""

    stroke_detected: bool = Field(description="True if stroke risk exceeds Youden threshold")
    probability: float = Field(description="Predicted probability of stroke P(stroke=1)")
    risk_level: str = Field(description='Risk level: "Bajo" or "Alto"')
    model_version: int = Field(description="MLflow model version")


model, version_model, data_dict = load_model(MODEL_NAME, ALIAS)

app = FastAPI(title="Stroke Prediction API", version="1.0.0")


@app.get("/")
async def read_root():
    """Health check endpoint."""
    return JSONResponse(
        content=jsonable_encoder({"message": "Welcome to the Stroke Prediction API"})
    )


@app.post("/predict/", response_model=ModelOutput)
def predict(
    features: Annotated[ModelInput, Body(embed=True)],
    background_tasks: BackgroundTasks,
):
    """Predict stroke risk from raw clinical features."""
    result = predict_with_threshold(model, features.model_dump(), data_dict)
    background_tasks.add_task(check_model)
    return ModelOutput(
        stroke_detected=result["stroke_detected"],
        probability=result["probability"],
        risk_level=result["risk_level"],
        model_version=version_model,
    )
