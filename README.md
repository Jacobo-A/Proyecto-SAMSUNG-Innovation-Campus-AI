# Proyecto SAMSUNG - Predicción de Calidad del Aire en México

Este proyecto tiene como objetivo analizar y predecir la calidad del aire utilizando datos de estaciones de monitoreo. El proceso incluye la imputación de datos faltantes, normalización, clustering, y modelos de regresión y clasificación para predecir contaminantes y estados de calidad del aire.

## Estructura del Proyecto

1. **Preprocesamiento de Datos**:
   - Carga y limpieza de datos.
   - Normalización de características.
   - Imputación de valores faltantes utilizando un autoencoder LSTM bidireccional.

2. **Análisis Exploratorio**:
   - Visualización de series temporales.
   - Matrices de correlación.
   - Distribución de contaminantes por estación.

3. **Modelado**:
   - **Regresión Temporal**: Predicción de contaminantes (PM2.5, PM10, NOx, O3, CO) utilizando un modelo LSTM.
   - **Clasificación**: Clustering con K-Means y predicción de estados de calidad del aire usando un ensamble de modelos (Random Forest, SVM, Regresión Logística).

4. **Pipeline de Predicción**:
   - Integración de los modelos para predecir contaminantes y estados futuros.
   - Visualización de resultados.

## Requisitos

- Python 3.8+
- Bibliotecas principales:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn`, `tensorflow`, `joblib`

### 1. Modelo de Imputación (Autoencoder Bidireccional LSTM)

**Propósito**:  
Completar valores faltantes en las series temporales de contaminantes atmosféricos.

**Arquitectura**:
- **Encoder**:
  - Capa Bidireccional LSTM (128 unidades) + BatchNormalization + Dropout (20%)
  - Capa Bidireccional LSTM (64 unidades) → Espacio latente (32 dimensiones)
- **Decoder**:
  - Capa RepeatVector para reconstrucción temporal
  - Capas Bidireccionales LSTM (64 y 128 unidades)
  - Capa de salida con activación sigmoide

**Características clave**:
- Uso de máscaras para ignorar valores NaN durante el entrenamiento
- Normalización por estación (MinMaxScaler)
- Entrenamiento con ventanas temporales de 7 días

### 2. Modelo de Regresión Temporal (LSTM)

**Propósito**:  
Predecir valores futuros de PM2.5, PM10, NOx, O3 y CO para 1 día adelante.

**Arquitectura**:
- **Entrada**: Secuencias de 7 días × 13 características
- **Capas**:
  - Bidireccional LSTM (128 unidades) → BatchNorm → Dropout
  - Bidireccional LSTM (64 unidades)
  - Capas Densas (64 y 32 unidades, ReLU)
- **Salida**: 5 unidades (contaminantes) con activación sigmoide

**Métricas**:  
Pérdida MSE (Mean Squared Error)

### 3. Modelo de Clasificación (Ensemble Stacking)

**Propósito**:  
Clasificar el estado de calidad del aire en clusters identificados por K-Means.

**Pipeline**:
1. **Preprocesamiento**:
   - Normalización (StandardScaler)
   - Reducción dimensional (PCA, 95% varianza)
2. **Modelos Base**:
   - Random Forest (optimizado con RandomizedSearchCV)
   - SVM (kernel RBF/linear, optimizado)
   - Regresión Logística
3. **Meta-modelo**:  
   Regresión Logística con regularización L2

**Optimización**:
- Búsqueda de hiperparámetros para cada modelo base
- Validación cruzada (5 folds)

### 4. Clustering (MiniBatchKMeans)

**Propósito**:  
Identificar patrones ocultos en los datos para crear etiquetas de clasificación.

**Implementación**:
- Selección automática de k (2-7 clusters) mediante Silhouette Score
- Visualización con PCA 2D

## Flujo de Predicción Integrado

1. **Entrada**: Últimos 7 días de datos observados
2. **Paso 1**: Imputación de faltantes (si aplica)
3. **Paso 2**: Predicción de contaminantes (Regresor LSTM)
4. **Paso 3**: 
   - Extracción de características cíclicas (día/semana/mes)
   - Clasificación del estado (Ensemble Stacking)
5. **Salida**:  
   - Valores predichos de contaminantes (denormalizados)
   - Etiqueta de cluster/estado

## Métricas de Evaluación

| Modelo          | Métrica Principal               | Otras Métricas                     |
|-----------------|----------------------------------|------------------------------------|
| Autoencoder     | MSE (reconstrucción)            | Pérdida de validación              |
| Regresor LSTM   | MSE (predicción 1 día)          | MAE, R²                            |
| Clasificador    | Accuracy                        | F1-score, Matriz de confusión      |
| Clustering      | Silhouette Score                | Inercia, Calinski-Harabasz         |

## Notas Técnicas

- **Normalización**:  
  Se aplica MinMaxScaler independiente por estación para mantener la consistencia espacial.

- **Manejo de Temporalidad**:  
  Las características cíclicas (día de semana, mes) se codifican mediante senos/cosenos para preservar su naturaleza circular.

- **Reproducibilidad**:  
  Semillas aleatorias fijas (42/43) en todos los modelos.

- **Persistencia**:  
  Todos los modelos y scalers se guardan en formato .h5 (Keras) y .pkl (scikit-learn) para reutilización.



## Uso

1. **Ejecución del Proyecto**:
   - Clona el repositorio y ejecuta el script `proyecto_samsung_.py`.
   - Asegúrate de tener los archivos de datos en la ruta especificada.

2. **Entrenamiento de Modelos**:
   - Los modelos se entrenan automáticamente y se guardan en la carpeta `Modelos_Imputadores_2`, `Modelos_Regresiones_2`, y `Modelos_Clasificadores_2`.

3. **Predicciones**:
   - El pipeline de predicción genera resultados para cada estación y los guarda en `df_imputed_full_denormalized.csv`.

## Resultados

- Gráficos de evolución temporal de contaminantes.
- Matrices de confusión para los modelos de clasificación.
- Predicciones diarias de calidad del aire para enero de 2022.

## Contribuciones

Contribuciones son bienvenidas. Por favor, abre un issue o envía un pull request con tus sugerencias.
