# StockifAI Forecast · Demanda de repuestos con LightGBM + Azure ML

Módulo de **forecasting de demanda de repuestos a 4 semanas**, extraído de StockifAI (proyecto final de Ingeniería en Sistemas, UTN FRBA) y rediseñado para entrenarse y desplegarse con **Azure Machine Learning**.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-2E8B57)
![Azure ML](https://img.shields.io/badge/Azure_ML-planificado-0078D4?logo=microsoftazure&logoColor=white)
![Estado](https://img.shields.io/badge/estado-exploración-orange)

> El producto completo (Django + Angular, alertas de quiebre y sobrestock, dashboard) vive en [StockifAI2](https://github.com/agtorres1/StockifAI2). Este repo toma su componente de pronóstico y lo rediseña.

---

## El problema

Un negocio de repuestos necesita saber **cuánto va a vender de cada pieza en las próximas 4 semanas** para reponer stock sin inmovilizar capital ni quedarse sin piezas.

La dificultad no es el algoritmo, son los datos: la demanda es **extremadamente intermitente**.

| | |
|---|---|
| Egresos registrados | 70.193 |
| SKUs | 3.901 |
| Período | 259 semanas (ago-2020 a jul-2025) |
| Filas SKU × semana (desde la primera venta, con ceros) | ~617 mil |
| Semanas con venta por SKU (mediana, en 5 años) | **2** |
| SKUs que venden en ≥25% de sus semanas | 124 |
| Concentración | el 10% de los SKUs explica el **91%** de las unidades |

Un modelo que prediga "cero" para todo acierta casi siempre y no sirve para nada. El diseño del proyecto gira alrededor de ese hecho.

---

## Enfoque

| | |
|---|---|
| **Unidad** | SKU × semana (depósitos agregados) |
| **Horizonte** | 4 semanas |
| **Segmentación** | Por patrón de demanda (frecuente / intermitente / esporádica) |
| **Modelo** | LightGBM **global** con el segmento como feature, comparado contra un modelo por segmento |
| **Estrategia multi-horizonte** | Directa (un modelo por t+1…t+4), comparada contra la recursiva del sistema original |
| **Baselines** | Predecir 0 · media móvil de 8 semanas |
| **Validación** | Temporal con origen rodante (rolling origin), nunca k-fold aleatorio |

### Sistema original vs. rediseño

| | StockifAI (v1) | stockifai-forecast (v2) |
|---|---|---|
| Modelo | Un LightGBM por segmento | LightGBM global, segmento como feature (comparado contra v1) |
| Multi-horizonte | Recursivo | Directo (comparado contra recursivo) |
| Función objetivo | MAE | A evaluar: Tweedie vs. MAE, siempre contra baselines |
| Validación | Un único corte: 4 semanas de validación y 4 de test | Origen rodante sobre varios cortes |
| Ejecución | `django-crontab` dentro del backend | Pipeline de Azure ML |

### Por qué estas decisiones

- **Modelo global y no uno por SKU:** la mayoría de los SKUs tiene muy pocas ventas para aprender algo propio. Un modelo global aprende patrones compartidos entre piezas.
- **Estrategia directa y no recursiva:** en la recursiva, cada predicción alimenta la siguiente. En series intermitentes el error se arrastra y las predicciones colapsan a cero hacia t+4.
- **Baselines obligatorios:** con esta intermitencia, una métrica aislada no dice nada. El modelo vale solo si le gana a reglas simples.

---

## Estado

| Etapa | Estado |
|---|---|
| Estructura del repo y entorno (conda, Python 3.11) | ✅ |
| Exploración: carga, tipos, faltantes, agregación de depósitos | 🟡 En curso |
| Segmentación de SKUs y features | ⏳ |
| Baselines y modelos (global vs. segmento, directo vs. recursivo) | ⏳ |
| Migración del código a `src/stockifai_forecast/` | ⏳ |
| Entrenamiento y despliegue en Azure ML | ⏳ Planificado |

---

## Estructura

```
data/raw/                 egresos históricos (datos de demo)
notebooks/                exploración y experimentos
src/stockifai_forecast/   paquete reutilizable: series, features, modelos, evaluación
tests/                    tests del paquete
requirements.txt
```

Lo que se valida en `notebooks/` se mueve a `src/stockifai_forecast/`, y ese mismo paquete es el que corre en Azure ML.

---

## Cómo reproducirlo

```bash
git clone https://github.com/agtorres1/stockifai-forecast.git
cd stockifai-forecast
conda create -n stockifai python=3.11
conda activate stockifai
pip install -r requirements.txt
```

Luego, abrir los notebooks de `notebooks/` en orden.

---

## Stack

Python 3.11 · pandas · LightGBM · scikit-learn · Azure Machine Learning (planificado)

---

**Autora:** Agustina Torres · Ingeniería en Sistemas, UTN FRBA
