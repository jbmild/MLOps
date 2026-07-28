# Roadmap del proyecto

## Entrega parcial (clase 5) — ~50%

- [x] Infra Docker (Airflow, MLflow, MinIO, Postgres, FastAPI)
- [x] Dataset seedeado en `data/seed/`
- [x] Documentación del modelo (`docs/MODELO.md`)
- [x] DAG `process_etl_stroke` → MinIO
- [x] Notebook HPO + registro `champion` en MLflow
- [x] README con demo parcial

## Entrega final

- [x] FastAPI con preprocessing alineado al ETL
- [x] DAG `retrain_stroke_model` (champion vs challenger)
- [x] `docs/ARQUITECTURA.md` + diagrama
- [x] Demo end-to-end documentada

## Demo entrega parcial

```bash
cp .env.example .env
docker compose --profile all up -d --build
# Airflow UI :8080 → DAG process_etl_stroke
# Notebook notebooks/experiment_mlflow.ipynb
# MLflow UI :5001 → stroke_prediction_model_prod alias champion
```

## Demo entrega final

Ver [README.md](../README.md#demo-end-to-end).
