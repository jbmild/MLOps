# Modelo: Predicción de ACV (Stroke)

Referencia completa del análisis: [ceia-ap-maquina/tp.ipynb](https://github.com/jbmild/ceia-ap-maquina/blob/main/tp.ipynb)

## Problema

Clasificación binaria para predecir si un paciente sufrirá un accidente cerebrovascular (`stroke`).
El dataset tiene ~5.110 registros con fuerte desbalance (~5% de casos positivos).

## Modelo elegido para producción

**Regresión Logística** con `GridSearchCV`, seleccionada por el equipo por:

- Mayor **recall** en test (~0.77) frente a alternativas con mejor F1 pero menor sensibilidad
- **ROC-AUC** ~0.84
- Interpretabilidad clínica de los coeficientes

> Bagging y Random Forest lograron F1 ligeramente superior en el benchmark, pero LR prioriza
> detectar más casos de ACV (screening clínico), trade-off documentado en el TP de AMq1.

### Hiperparámetros óptimos (GridSearchCV, 5-fold CV, métrica F1)

| Parámetro | Valor |
|-----------|-------|
| `C` | 0.1 |
| `penalty` | l1 |
| `solver` | liblinear |
| `class_weight` | balanced |
| F1 CV | 0.223 |

### Umbral de decisión

Umbral **Youden** = **0.441**, calculado por validación cruzada sobre el conjunto de entrenamiento
(no sobre test), para maximizar TPR − FPR.

## Preprocesamiento

1. Eliminar columna `id`
2. Excluir registros con `gender = Other` (1 fila)
3. `ever_married`: Yes → 1, No → 0
4. Feature engineering clínico:
   - `bmi_category`: bajo_peso / normal / sobrepeso / obeso
   - `glucose_category`: bajo / normal / prediabetes / diabetes
   - `age_group`: <18 / 18-35 / 36-50 / 51-65 / >65
5. One-hot encoding (`drop_first=True`) sobre 7 categóricas
6. Split estratificado 70/30 (`random_state=42`)
7. Pipeline sklearn: imputación mediana + `StandardScaler` en numéricas; passthrough en dummies

## Métricas en test (umbral Youden)

| Métrica | Valor |
|---------|-------|
| Recall (ACV) | ~0.77 |
| F1 (ACV) | ~0.21 |
| ROC-AUC | ~0.84 |
| Accuracy | ~0.71 |
