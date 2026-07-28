# Roadmap del proyecto

## Entrega parcial (clase 5) — ~50%

- [x] Infra Docker (Airflow, MLflow, MinIO, Postgres, FastAPI)
- [x] Dataset seedeado en `data/seed/`
- [x] Documentación del modelo (`docs/MODELO.md`)
- [ ] DAG `process_etl_stroke` → MinIO
- [ ] Notebook HPO + registro `champion` en MLflow
- [ ] README con demo parcial

## Entrega final

- [ ] FastAPI con preprocessing alineado al ETL
- [ ] DAG `retrain_stroke_model` (champion vs challenger)
- [ ] `docs/ARQUITECTURA.md` + diagrama
- [ ] Demo end-to-end documentada

## Demo entrega parcial

```bash
cp .env.example .env   # ajustar AIRFLOW_UID si hace falta
docker compose --profile all up -d
# Airflow UI :8080 → ejecutar DAG process_etl_stroke
# Notebook notebooks/experiment_mlflow.ipynb
# MLflow UI :5001 → modelo stroke_prediction_model_prod alias champion
```

## Demo entrega final

```bash
# Tras ETL + notebook HPO:
curl -X POST http://localhost:8800/predict/ \
  -H 'Content-Type: application/json' \
  -d '{"features": {"gender": "Male", "age": 67, "hypertension": 0, "heart_disease": 1,
       "ever_married": 1, "work_type": "Private", "Residence_type": "Urban",
       "avg_glucose_level": 228.69, "bmi": 36.6, "smoking_status": "formerly smoked"}}'
# Airflow → DAG retrain_stroke_model
```
