# Stroke Prediction — MLOps CEIA

Predicción de accidente cerebrovascular (ACV) en producción usando el modelo de
[Aprendizaje de Máquina I (ceia-ap-maquina)](https://github.com/jbmild/ceia-ap-maquina).

**Stack:** Apache Airflow · MLflow · MinIO · PostgreSQL · FastAPI · Docker

## Modelo

- **Algoritmo:** Regresión Logística + GridSearchCV (`C=0.1`, `penalty=l1`, `class_weight=balanced`)
- **Umbral:** Youden = 0.441 (validación cruzada sobre train)
- **Detalle:** [docs/MODELO.md](docs/MODELO.md)

## Requisitos

- Docker + Docker Compose
- ~4 GB RAM
- Linux: configurar `AIRFLOW_UID` en `.env` (ver `.env.example`)

## Levantar el stack

```bash
cp .env.example .env
# En Linux: id -u  →  AIRFLOW_UID=<tu_uid>
docker compose --profile all up -d --build
```

| Servicio | URL |
|----------|-----|
| Airflow | http://localhost:8080 (airflow / airflow) |
| MLflow | http://localhost:5001 |
| MinIO Console | http://localhost:9001 (minio / minio123) |
| API | http://localhost:8800/docs |

## Demo end-to-end

### 1. ETL (Airflow)

En Airflow UI, ejecutar el DAG **`process_etl_stroke`**.

Genera en MinIO (`s3://data`):
- `final/train/stroke_X_train.csv`, `stroke_y_train.csv`
- `final/test/stroke_X_test.csv`, `stroke_y_test.csv`
- `data_info/data.json`

### 2. Entrenamiento + registro (Notebook)

```bash
# Instalar deps locales para el notebook (opcional)
python -m venv .venv && .venv/bin/pip install awswrangler mlflow scikit-learn pandas boto3
```

Abrir y ejecutar [`notebooks/experiment_mlflow.ipynb`](notebooks/experiment_mlflow.ipynb).

Registra `stroke_prediction_model_prod` con alias **`champion`** en MLflow.

### 3. Predicción (API)

```bash
curl -X POST http://localhost:8800/predict/ \
  -H 'Content-Type: application/json' \
  -d '{
    "features": {
      "gender": "Male",
      "age": 67,
      "hypertension": 0,
      "heart_disease": 1,
      "ever_married": 1,
      "work_type": "Private",
      "Residence_type": "Urban",
      "avg_glucose_level": 228.69,
      "bmi": 36.6,
      "smoking_status": "formerly smoked"
    }
  }'
```

Respuesta esperada:

```json
{
  "stroke_detected": true,
  "probability": 0.7974,
  "risk_level": "Alto",
  "model_version": 1
}
```

### 4. Reentrenamiento (Airflow)

Ejecutar de nuevo `process_etl_stroke` (opcional, para datos frescos) y luego el DAG **`retrain_stroke_model`**.

## Estructura del repo

```
├── airflow/dags/          # DAGs ETL y retrain
├── data/seed/             # CSV original (sin Kaggle en prod)
├── dockerfiles/           # Imágenes Docker
├── docs/                  # Documentación del modelo y arquitectura
├── notebooks/             # HPO + registro MLflow
└── src/                   # features.py, train.py, inference.py
```

## Documentación

- [docs/MODELO.md](docs/MODELO.md) — decisiones de modelado del TP AMq1
- [docs/ARQUITECTURA.md](docs/ARQUITECTURA.md) — diagrama y flujo de datos
- [docs/ROADMAP.md](docs/ROADMAP.md) — estado del proyecto

## Integrantes

Grupo del **Trabajo Práctico de Operaciones de Aprendizaje Automático I** — CEIA, FIUBA.

- Jonatan Mild
- Valentin Torres
- Ignacio	Vollono Cadenazzi

_Integrantes adicionales: pendiente de completar._

El modelo desplegado proviene del TP de Aprendizaje de Máquina I (equipo distinto):
[ceia-ap-maquina](https://github.com/jbmild/ceia-ap-maquina).

## Licencia

Ver [LICENSE](LICENSE).
