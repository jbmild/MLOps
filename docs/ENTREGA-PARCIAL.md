# TP MLOps I — Entrega parcial

**Integrantes:** Jonatan Mild, Valentín Torres, Ignacio Vollono Cadenazzi

**Repo:** https://github.com/jbmild/MLOps

Hola! Va nuestro avance del TP. Fuimos por el **nivel en contenedores**, poniendo en producción el
modelo de predicción de ACV (regresión logística + GridSearchCV) que habíamos armado en Aprendizaje
de Máquina I, repo: https://github.com/jbmild/ceia-ap-maquina

## Avance hasta hoy

- **`airflow/dags/process_etl_stroke.py`**: el ETL orquestado con TaskFlow. Levanta el dataset,
  hace la limpieza y el feature engineering clínico, y deja el split estratificado en
  `s3://data/final/`. Los datos van y vienen por S3, por XCom mandamos solo los punteros.

- **`airflow/dags/train_stroke_model.py`**: corre la búsqueda de hiperparámetros y registra todo en
  MLflow (parámetros, métricas, la matriz de confusión y la curva ROC), y le pone el alias
  `champion` a la versión ganadora.

- **`airflow/dags/retrain_stroke_model.py`**: reentrena y compara el challenger contra el champion.
  Si el challenger gana lo promueve, y al anterior le deja el alias `old_champion` por si hay que
  volver atrás.

- **`src/`**: este es el punto que más nos importó cuidar. Es el mismo código de preprocesamiento
  que usan el ETL, el entrenamiento y la API, así que no tenemos dos implementaciones que se puedan
  ir desincronizando con el tiempo.

- **`docs/`**: dejamos capturas de una corrida completa hecha dentro de Docker, así se puede ver el
  resultado sin tener que levantar todo el stack.

## Cómo pensamos seguir

- **Predicción en lote.** Por ahora solo tenemos inferencia online contra la API. La idea es sumar
  un DAG que corra cada tanto, precompute las predicciones de toda la cohorte de pacientes y las
  deje guardadas en Redis, armando la clave a partir de las features. De esa forma la consulta pasa
  a ser una lectura de base y no una inferencia en tiempo real. Como mencionaste en una de las
  clases, para la mayoría de los casos con batch alcanza y te ahorra un montón de infraestructura,
  así que queremos tener las dos variantes para poder compararlas.

- **Seguridad en la API.** Hoy el endpoint está abierto y las categóricas entran como string libre,
  así que si te mandan un `work_type` que no existe el modelo igual devuelve una predicción (los
  dummies quedan todos en cero y no salta ningún error). Vamos a agregar autenticación y validar
  las categorías contra los valores válidos que ya tenemos guardados en `data.json`.

- **Champion vs challenger.** El challenger todavía reentrena con los mismos hiperparámetros del
  champion, nos falta que haga su propia búsqueda. Y la comparación entre los dos la hacemos con el
  umbral 0.5 por defecto, cuando la API sirve con el umbral de Youden, así que estamos midiendo
  algo distinto de lo que realmente está en producción.

- **Tests** del preprocesamiento y un smoke test punta a punta antes de promover un modelo.

## Una consulta

Para el nivel en contenedores, ¿esperás que cubramos serving online y predicción en lote, o con uno
de los dos bien resuelto alcanza?

Muchas gracias!
