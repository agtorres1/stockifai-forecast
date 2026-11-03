# StockifAI Forecast · Demanda de repuestos con LightGBM + Azure ML

Módulo de **forecasting de demanda de repuestos a 4 semanas**, extraído de StockifAI (proyecto final de Ingeniería en Sistemas, UTN FRBA) y rediseñado para entrenarse y desplegarse con **Azure Machine Learning**.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-2E8B57)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)
![Tests](https://img.shields.io/badge/tests-57_passing-success)
![Azure ML](https://img.shields.io/badge/Azure_ML-planificado-0078D4?logo=microsoftazure&logoColor=white)
![Estado](https://img.shields.io/badge/estado-modelo_evaluado-yellowgreen)

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
| Filas SKU × semana (desde la primera venta, con ceros) | 616.866 |
| Semanas con venta por SKU (mediana, en 5 años) | **2** |
| SKUs que venden en ≥25% de sus semanas | 124 |
| Concentración | el 10% de los SKUs explica el **91%** de las unidades |

Un modelo que prediga "cero" para todo acierta casi siempre y no sirve para nada. El diseño del proyecto gira alrededor de ese hecho.

---

## Resultados

Evaluación con **origen rodante**: 4 exámenes de 26 semanas que no se pisan (desde 31/07/2023, 29/01/2024, 29/07/2024 y 27/01/2025). En cada uno se recalcula todo con su propio corte: universo, segmentos, baselines y modelo.

MAE promedio entre exámenes y horizontes (unidades por semana, menor es mejor):

| Grupo | Predecir 0 | Media móvil 8 sem. | **LightGBM** | LightGBM gana |
|---|---|---|---|---|
| Frecuencia alta | 3,370 | 1,534 | **1,375** | 4 de 4 |
| Historia ≥26 semanas con venta | 2,066 | 1,013 | **0,873** | 4 de 4 |
| Historia 10–25 semanas con venta | **0,097** | 0,165 | 0,098 | 1 de 4 |
| Intermitentes | **0,112** | 0,191 | 0,113 | 0 de 4 |
| **Total** | 0,940 | 0,529 | **0,430** | **4 de 4** |

**Lectura:**

- En los repuestos que se venden seguido, LightGBM le gana a las reglas simples **en los 4 períodos**. En el período más reciente, el error total baja un 22 % frente a la media móvil.
- En los intermitentes, el modelo no mejora a "predecir 0". Ahí se usa una regla simple.
- En los repuestos con 10 a 25 semanas con venta, el modelo gana solo 1 de 4 veces, así que el universo del modelo pasa a **≥26 semanas con venta**.
- Variables más útiles para t+1: `prom_12`, `semanas_con_venta_8` y `cantidad`.

> Que "predecir 0" gane en MAE no la vuelve buena para reponer: nunca repone nada. El MAE no mide el costo de quedarse sin stock (ver *Limitaciones*).

---

## Enfoque

| | |
|---|---|
| **Unidad** | SKU × semana (depósitos agregados, semanas de lunes a domingo) |
| **Fila de la ABT** | Un SKU en una semana recién cerrada: mira esa semana y las anteriores, y predice las 4 siguientes |
| **Horizonte** | 4 semanas |
| **Universo del modelo** | SKUs con ≥26 semanas con venta, contadas solo con datos de entrenamiento (decidido con el backtesting, que se corrió con el corte anterior de 10). El resto recibe una regla simple |
| **Segmentación** | Frecuencia alta (vende en ≥25% de sus semanas) / intermitente, usada como feature y para cortar la evaluación |
| **Modelo** | LightGBM **global** (un modelo para todos los SKUs), objetivo L1 |
| **Estrategia multi-horizonte** | Directa: 4 modelos, uno por cada semana adelante (t+1…t+4) |
| **Features** | Venta de la semana, lags 1–3, medias móviles 4/8/12, desvío y semanas con venta en las últimas 8, semanas desde la última venta, mes, semana del año, feriados nacionales de las 4 semanas a predecir |
| **Baselines** | Predecir 0 · media móvil de 8 semanas |
| **Validación** | Corte temporal con hueco + validación interna para early stopping + backtesting de 4 exámenes. Nunca k-fold aleatorio |

### Cómo se evita la fuga de datos

- **Features que solo miran hacia atrás**, calculadas por SKU (`groupby`), para que un repuesto nunca use datos de otro.
- **Hueco de 4 semanas** entre entrenamiento y test: se descartan las filas de entrenamiento cuyos targets caerían en el test. Lo mismo entre fit y validación.
- **Universo y segmentos calculados solo con historia anterior al corte.** Con el corte de 10 semanas, contando todo el período entraban 494 SKUs; con la historia correcta, 456. Esos 38 calificaban gracias a ventas del test.
- **Early stopping sobre una validación propia**, nunca sobre el test.
- **Una sola función de features** para entrenamiento e inferencia.
- Tests automáticos para cada una de estas reglas.

### Sistema original vs. rediseño

| | StockifAI (v1) | stockifai-forecast (v2) |
|---|---|---|
| Modelo | Un LightGBM por segmento | LightGBM global, segmento como feature |
| Multi-horizonte | Recursivo | Directo (4 modelos) |
| Función objetivo | MAE | L1 (Tweedie evaluado: no mejoró el MAE) |
| Validación | Un único corte; early stopping sobre la semana evaluada | Hueco temporal, validación propia y backtesting de 4 exámenes |
| Features en inferencia | Una sola fila por SKU (lags largos vacíos) | Misma función de features que en entrenamiento |
| Medias móviles | Sin separar por SKU | Por SKU |
| Ejecución | `django-crontab` dentro del backend | Pipeline de Azure ML (planificado) |

---

## Estado

| Etapa | Estado |
|---|---|
| Estructura del repo y entorno (conda, Python 3.11) | ✅ |
| Exploración y decisiones de datos | ✅ |
| Módulo de datos: lectura, agregación semanal, ceros, etiquetas | ✅ |
| ABT: features y targets sin fuga | ✅ |
| Baselines y corte temporal con hueco | ✅ |
| LightGBM directo, L1 vs. Tweedie, MLflow | ✅ |
| Backtesting con origen rodante | ✅ |
| Predicción de las próximas 4 semanas con todos los datos | ⏳ Próximo paso |
| Entrenamiento y despliegue en Azure ML | ⏳ Planificado |

---

## Estructura

```
data/raw/                     egresos históricos (datos de demo)
notebooks/
  01_exploracion.ipynb        exploración y decisiones de datos
  02_pruebas_modulo.ipynb     validación del módulo de datos
  03_features.ipynb           construcción de la ABT
  04_baselines.ipynb          corte temporal y baselines
  05_lightgbm.ipynb           entrenamiento y comparación
  06_backtesting.ipynb        evaluación en 4 períodos
src/stockifai_forecast/
  datos.py                    lectura, agregación semanal, ceros, etiquetas
  features.py                 ABT: features y targets
  evaluacion.py               corte con hueco, etiquetas con train, baselines
  modelado.py                 validación, entrenamiento, predicción, métricas
  backtesting.py              evaluación con origen rodante
tests/                        57 tests (incluye controles anti-fuga)
requirements.txt
```

Lo que se valida en `notebooks/` vive en `src/stockifai_forecast/`, y ese mismo paquete es el que va a correr en Azure ML.

---

## Cómo correrlo

```bash
conda create -n stockifai python=3.11
conda activate stockifai
pip install -r requirements.txt

pytest                         # 57 tests
jupyter notebook notebooks/    # notebooks en orden, del 01 al 06
```

Los experimentos se registran en MLflow local (`mlruns/`, fuera de Git): `mlflow ui` para verlos.

---

## Limitaciones

- **Demanda censurada:** si no había stock, no hay salida registrada. El modelo ve un 0 que puede no ser falta de demanda.
- **El MAE favorece predecir 0** en los intermitentes. Falta una métrica que castigue quedarse sin stock.
- **Corte del universo en 26:** respaldado por 4 períodos históricos, pero conviene confirmarlo con un período futuro.
- **Feriados:** se usan los nacionales; pueden faltar feriados puente.
- **Hiperparámetros sin ajustar**: solo la cantidad de árboles, con early stopping.

---

## Próximos pasos

- Reentrenar con todos los datos y predecir las próximas 4 semanas de cada SKU.
- Elegir la regla para los SKUs fuera del modelo y para los intermitentes.
- Ajustar hiperparámetros contra la validación.
- Comparar el modelo global contra uno por segmento, y el horizonte directo contra el recursivo.
- Validación de datos en cada ingesta y tests automáticos con GitHub Actions.
- Pipeline en Azure ML: Data Asset, entrenamiento, gate de calidad y batch endpoint semanal.
