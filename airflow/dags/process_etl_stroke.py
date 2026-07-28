import datetime
import json
import os

from airflow.decorators import dag, task

markdown_text = """
### ETL Process for Stroke Prediction Data

This DAG loads the stroke dataset from MinIO (seeded from `data/seed/`), applies the same
preprocessing as the AMq1 notebook (cleaning, clinical feature engineering, one-hot encoding),
splits into train/test (70/30 stratified), and saves CSVs plus metadata to `s3://data`.
"""


default_args = {
    "owner": "MLOps CEIA",
    "depends_on_past": False,
    "schedule_interval": None,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
    "dagrun_timeout": datetime.timedelta(minutes=15),
}


@dag(
    dag_id="process_etl_stroke",
    description="ETL for stroke data: preprocessing, split and upload to MinIO.",
    doc_md=markdown_text,
    tags=["ETL", "Stroke"],
    default_args=default_args,
    catchup=False,
)
def process_etl_stroke():

    @task
    def upload_raw_data():
        """Upload seed CSV to MinIO raw bucket."""
        import awswrangler as wr
        import pandas as pd

        seed_path = "/opt/airflow/data/seed/healthcare-dataset-stroke-data.csv"
        if not os.path.exists(seed_path):
            raise FileNotFoundError(f"Seed CSV not found at {seed_path}")

        df = pd.read_csv(seed_path)
        wr.s3.to_csv(df=df, path="s3://data/raw/stroke.csv", index=False)

    @task
    def preprocess_and_split():
        """Preprocess data and split into train/test sets."""
        import datetime as dt

        import awswrangler as wr
        import mlflow
        import pandas as pd
        from airflow.models import Variable
        from sklearn.model_selection import train_test_split

        from features import (
            TARGET_COL,
            add_clinical_features,
            build_data_metadata,
            clean_raw_dataframe,
            encode_features,
            get_feature_columns,
        )

        dataset = wr.s3.read_csv("s3://data/raw/stroke.csv")
        df_clean = clean_raw_dataframe(dataset)
        df_fe = add_clinical_features(df_clean)
        df_encoded = encode_features(df_fe)

        test_size = float(Variable.get("test_size_stroke", default_var="0.3"))
        random_state = int(Variable.get("random_state_stroke", default_var="42"))

        X = df_encoded.drop(columns=TARGET_COL)
        y = df_encoded[[TARGET_COL]]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        wr.s3.to_csv(X_train, "s3://data/final/train/stroke_X_train.csv", index=False)
        wr.s3.to_csv(X_test, "s3://data/final/test/stroke_X_test.csv", index=False)
        wr.s3.to_csv(y_train, "s3://data/final/train/stroke_y_train.csv", index=False)
        wr.s3.to_csv(y_test, "s3://data/final/test/stroke_y_test.csv", index=False)

        metadata = build_data_metadata(df_fe, df_encoded)
        metadata["feature_columns"] = get_feature_columns(df_encoded)
        metadata["date"] = dt.datetime.today().strftime("%Y/%m/%d-%H:%M:%S")

        import boto3
        client = boto3.client("s3")
        client.put_object(
            Bucket="data",
            Key="data_info/data.json",
            Body=json.dumps(metadata, indent=2),
        )

        mlflow.set_tracking_uri("http://mlflow:5000")
        experiment = mlflow.set_experiment("Stroke Prediction")
        mlflow.start_run(
            run_name="ETL_run_" + dt.datetime.today().strftime("%Y/%m/%d-%H:%M:%S"),
            experiment_id=experiment.experiment_id,
            tags={"experiment": "etl", "dataset": "stroke"},
        )
        mlflow.log_param("train_observations", X_train.shape[0])
        mlflow.log_param("test_observations", X_test.shape[0])
        mlflow.log_param("features", len(get_feature_columns(df_encoded)))
        mlflow.end_run()

    upload_raw_data() >> preprocess_and_split()


dag = process_etl_stroke()
