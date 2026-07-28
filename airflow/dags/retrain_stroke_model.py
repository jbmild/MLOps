import datetime

from airflow.decorators import dag, task

markdown_text = """
### Re-Train Stroke Prediction Model

This DAG re-trains the stroke model on current MinIO data, registers a challenger,
compares F1 on the test set against the champion, and promotes the challenger only if better.
"""


default_args = {
    "owner": "MLOps CEIA",
    "depends_on_past": False,
    "schedule_interval": None,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
    "dagrun_timeout": datetime.timedelta(minutes=20),
}


@dag(
    dag_id="retrain_stroke_model",
    description="Retrain stroke model with champion vs challenger comparison.",
    doc_md=markdown_text,
    tags=["Re-Train", "Stroke"],
    default_args=default_args,
    catchup=False,
)
def retrain_stroke_model():

    @task
    def train_challenger():
        import datetime as dt

        import awswrangler as wr
        import mlflow
        from mlflow.models import infer_signature
        from sklearn.base import clone
        from sklearn.metrics import f1_score

        from train import MODEL_NAME

        mlflow.set_tracking_uri("http://mlflow:5000")
        client = mlflow.MlflowClient()

        champion = mlflow.sklearn.load_model(
            client.get_model_version_by_alias(MODEL_NAME, "champion").source
        )
        challenger = clone(champion)

        X_train = wr.s3.read_csv("s3://data/final/train/stroke_X_train.csv")
        y_train = wr.s3.read_csv("s3://data/final/train/stroke_y_train.csv").squeeze()
        X_test = wr.s3.read_csv("s3://data/final/test/stroke_X_test.csv")
        y_test = wr.s3.read_csv("s3://data/final/test/stroke_y_test.csv").squeeze()

        challenger.fit(X_train, y_train)
        y_pred = challenger.predict(X_test)
        f1 = f1_score(y_test, y_pred)

        experiment = mlflow.set_experiment("Stroke Prediction")
        mlflow.start_run(
            run_name="Challenger_run_" + dt.datetime.today().strftime("%Y/%m/%d-%H:%M:%S"),
            experiment_id=experiment.experiment_id,
            tags={"experiment": "challenger", "dataset": "stroke"},
        )
        mlflow.log_params(challenger.get_params())
        mlflow.log_metric("test_f1", f1)

        signature = infer_signature(X_train, challenger.predict_proba(X_train))
        mlflow.sklearn.log_model(
            sk_model=challenger,
            artifact_path="model",
            signature=signature,
            registered_model_name=MODEL_NAME,
        )
        model_uri = mlflow.get_artifact_uri("model")
        mlflow.end_run()

        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        latest = max(int(v.version) for v in versions)
        client.set_registered_model_alias(MODEL_NAME, "challenger", str(latest))

        return {"f1_challenger": f1, "challenger_version": latest}

    @task
    def evaluate_and_promote(train_result: dict):
        import mlflow
        import awswrangler as wr
        from sklearn.metrics import f1_score

        from train import MODEL_NAME

        mlflow.set_tracking_uri("http://mlflow:5000")
        client = mlflow.MlflowClient()

        champion = mlflow.sklearn.load_model(
            client.get_model_version_by_alias(MODEL_NAME, "champion").source
        )
        challenger = mlflow.sklearn.load_model(
            client.get_model_version_by_alias(MODEL_NAME, "challenger").source
        )

        X_test = wr.s3.read_csv("s3://data/final/test/stroke_X_test.csv")
        y_test = wr.s3.read_csv("s3://data/final/test/stroke_y_test.csv").squeeze()

        f1_champion = f1_score(y_test, champion.predict(X_test))
        f1_challenger = f1_score(y_test, challenger.predict(X_test))

        experiment = mlflow.set_experiment("Stroke Prediction")
        runs = mlflow.search_runs([experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
        if not runs.empty:
            with mlflow.start_run(run_id=runs.iloc[0]["run_id"]):
                mlflow.log_metric("test_f1_champion", f1_champion)
                mlflow.log_metric("test_f1_challenger", f1_challenger)
                winner = "Challenger" if f1_challenger > f1_champion else "Champion"
                mlflow.set_tag("winner", winner)

        if f1_challenger > f1_champion:
            challenger_version = client.get_model_version_by_alias(MODEL_NAME, "challenger")
            client.delete_registered_model_alias(MODEL_NAME, "champion")
            client.delete_registered_model_alias(MODEL_NAME, "challenger")
            client.set_registered_model_alias(
                MODEL_NAME, "champion", challenger_version.version
            )
        else:
            client.delete_registered_model_alias(MODEL_NAME, "challenger")

        return {
            "f1_champion": f1_champion,
            "f1_challenger": f1_challenger,
            "promoted": f1_challenger > f1_champion,
        }

    result = train_challenger()
    evaluate_and_promote(result)


dag = retrain_stroke_model()
