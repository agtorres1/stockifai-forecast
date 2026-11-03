import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from stockifai_forecast.evaluacion import evaluar_baselines


_FEATURES = [
    "cantidad", "lag_1", "lag_2", "lag_3", "prom_4", "prom_8", "prom_12",
    "desvio_8", "semanas_con_venta_8", "semanas_desde_ultima_venta",
    "mes", "semana_del_anio", "feriados_t1", "feriados_t2", "feriados_t3",
    "feriados_t4", "segmento",
]
_SEGMENTOS = ["frecuencia_alta", "intermitente"]


def columnas_features(abt: pd.DataFrame) -> list[str]:
    """Lista explícita de 16 features más segmento; nunca infiere por descarte.

    Espera la ABT predeterminada (ventanas 4/8/12, calendario de 4 horizontes)
    con segmento unido. Una feature ausente produce un error, no se omite.
    """
    faltantes = [c for c in _FEATURES if c not in abt.columns]
    if faltantes:
        raise ValueError(f"Faltan features: {faltantes}")
    return _FEATURES.copy()


def separar_validacion(
    train: pd.DataFrame, semanas_val: int = 8, horizonte: int = 4
) -> dict:
    """Reserva los últimos orígenes de train y purga fit sin reiniciar índices."""
    if not isinstance(semanas_val, int) or semanas_val < 1:
        raise ValueError("semanas_val debe ser un entero positivo")
    if not isinstance(horizonte, int) or horizonte < 1:
        raise ValueError("horizonte debe ser un entero positivo")
    if train.empty:
        raise ValueError("train no puede estar vacío")
    inicio_val = train["semana"].max() - pd.Timedelta(weeks=semanas_val - 1)
    fit = train.loc[
        train["semana"] + pd.Timedelta(weeks=horizonte) < inicio_val
    ].copy()
    val = train.loc[train["semana"] >= inicio_val].copy()
    return {"fit": fit, "val": val, "inicio_val": inicio_val}


def _preparar_x(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if "segmento" not in features or any(c not in _FEATURES for c in features):
        raise ValueError("features debe incluir segmento y solo columnas predictoras permitidas")
    if len(features) != len(set(features)):
        raise ValueError("features contiene columnas duplicadas")
    x = df.loc[:, features].copy()
    if x["segmento"].isna().any() or not x["segmento"].isin(_SEGMENTOS).all():
        raise ValueError("segmento debe tener una etiqueta conocida en cada fila")
    # Diccionario fijo, igual en fit, val y predicción, sin aprender del futuro.
    x["segmento"] = pd.Categorical(x["segmento"], categories=_SEGMENTOS)
    return x


def entrenar_modelos(
    fit: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    objetivo: str,
    horizonte: int = 4,
    semilla: int = 42,
) -> dict:
    """Entrena un modelo directo por horizonte, solo en el universo.

    No recibe test. Early stopping observa únicamente l1 en val. Las etiquetas
    deben calcularse aguas arriba con historia anterior a inicio_val.
    Las filas sin target se excluyen por horizonte; NaN en X se conservan.
    """
    if objetivo not in ("l1", "tweedie"):
        raise ValueError("objetivo debe ser l1 o tweedie")
    if not isinstance(horizonte, int) or horizonte < 1:
        raise ValueError("horizonte debe ser un entero positivo")
    fit = fit.loc[fit["entra_modelo"].eq(True).fillna(False)].copy()
    val = val.loc[val["entra_modelo"].eq(True).fillna(False)].copy()
    if fit.empty or val.empty:
        raise ValueError("fit y val deben contener filas del universo")
    if fit["semana"].max() + pd.Timedelta(weeks=horizonte) >= val["semana"].min():
        raise ValueError("los targets de fit deben quedar antes de validación")
    x_fit, x_val = _preparar_x(fit, features), _preparar_x(val, features)
    modelos = {}
    for h in range(1, horizonte + 1):
        target = f"target_{h}"
        conocidas_fit, conocidas_val = fit[target].notna(), val[target].notna()
        y_fit, y_val = fit.loc[conocidas_fit, target], val.loc[conocidas_val, target]
        if y_fit.empty or y_val.empty:
            raise ValueError(f"No hay respuestas conocidas para {target} en fit o val")
        if not np.isfinite(y_fit).all() or not np.isfinite(y_val).all():
            raise ValueError("Los targets deben ser finitos")
        if objetivo == "tweedie" and (
            (y_fit < 0).any() or (y_val < 0).any() or y_fit.sum() <= 0
        ):
            raise ValueError("Tweedie requiere targets no negativos y alguna venta en fit")
        parametros = dict(
            objective=objetivo, learning_rate=0.05, n_estimators=2000,
            random_state=semilla, metric="l1",
        )
        if objetivo == "tweedie":
            parametros["tweedie_variance_power"] = 1.5
        modelo = LGBMRegressor(**parametros)
        modelo.fit(
            x_fit.loc[conocidas_fit], y_fit,
            eval_X=x_val.loc[conocidas_val], eval_y=y_val,
            eval_metric="l1", categorical_feature=["segmento"],
            callbacks=[early_stopping(100, first_metric_only=True, verbose=False),
                       log_evaluation(0)],
        )
        modelos[h] = modelo
    return modelos


def predecir(modelos: dict, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Devuelve pred_h en el mismo índice y orden que df, recortadas a cero."""
    x = _preparar_x(df, features)
    predicciones = pd.DataFrame(index=df.index)
    for h, modelo in sorted(modelos.items()):
        if modelo.feature_name_ != features:
            raise ValueError("features debe coincidir con el orden usado al entrenar")
        predicciones[f"pred_{h}"] = (
            np.maximum(modelo.predict(x), 0) if len(x) else pd.Series(index=x.index, dtype=float)
        )
    return predicciones


def evaluar_modelo(
    test_etiquetado: pd.DataFrame, predicciones: pd.DataFrame, nombre: str
) -> pd.DataFrame:
    """Evalúa en las mismas claves de test, con el formato de evaluar_baselines.

    Exige índices únicos y exactamente las mismas filas; alinea por índice si
    las predicciones vienen reordenadas. No permite descartar silenciosamente
    filas del universo por predicciones ausentes. prom_8 no filtra al modelo:
    sus conteos se comparan con cero y se verifica la cobertura de prom_8 aparte.
    """
    if not test_etiquetado.index.is_unique or not predicciones.index.is_unique:
        raise ValueError("test y predicciones deben tener índices únicos")
    if len(test_etiquetado) != len(predicciones) or not (
        test_etiquetado.index.difference(predicciones.index).empty
        and predicciones.index.difference(test_etiquetado.index).empty
    ):
        raise ValueError("predicciones debe tener exactamente las mismas filas de test")
    columnas = list(predicciones.columns)
    if not columnas or set(columnas) != {f"pred_{h}" for h in range(1, len(columnas) + 1)}:
        raise ValueError("Se esperan pred_1 a pred_h sin saltos ni otras columnas")
    predicciones = predicciones.reindex(test_etiquetado.index)
    universo = test_etiquetado["entra_modelo"].eq(True).fillna(False)
    resultados = []
    for h in range(1, len(columnas) + 1):
        target, pred = test_etiquetado[f"target_{h}"], predicciones[f"pred_{h}"]
        evaluables = universo & target.notna()
        if not np.isfinite(pred.loc[evaluables]).all():
            raise ValueError("Faltan predicciones finitas para filas evaluables del universo")
        temporal = test_etiquetado.copy()
        temporal["prom_8"] = pred
        temporal["target_1"] = target
        # Reutiliza exactamente el cálculo y los grupos de los baselines.
        metricas = evaluar_baselines(temporal, horizonte=1)
        metricas = metricas.loc[metricas["baseline"] == "prom_8"].copy()
        metricas["baseline"], metricas["horizonte"] = nombre, h
        resultados.append(metricas)
    return pd.concat(resultados, ignore_index=True)
