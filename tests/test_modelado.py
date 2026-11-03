import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import stockifai_forecast.modelado as modelado
from stockifai_forecast.evaluacion import dividir_por_fecha, etiquetar_con_train, evaluar_baselines
from stockifai_forecast.features import construir_abt
from stockifai_forecast.modelado import (
    columnas_features, separar_validacion, entrenar_modelos, predecir, evaluar_modelo,
)


def serie_de_ejemplo():
    semanas = pd.date_range("2023-01-02", periods=100, freq="W-MON")
    return pd.DataFrame({
        "repuesto": ["A"] * 100 + ["B"] * 100,
        "semana": list(semanas) * 2,
        "cantidad": [i % 5 for i in range(100)]
        + [0 if i % 3 else 2 + i % 7 for i in range(100)],
    })


def tablas_de_ejemplo():
    serie = serie_de_ejemplo()
    abt = construir_abt(serie)
    partes = dividir_por_fecha(abt)
    internas = separar_validacion(partes["train"])
    etiquetas = etiquetar_con_train(serie, internas["inicio_val"])
    # join mantiene los índices de la ABT original para probar disjunción.
    fit = internas["fit"].join(etiquetas.set_index("repuesto"), on="repuesto")
    val = internas["val"].join(etiquetas.set_index("repuesto"), on="repuesto")
    return partes, internas, fit, val


@pytest.fixture
def modelos_chicos(monkeypatch):
    real = modelado.LGBMRegressor
    parametros = []

    def constructor(**kwargs):
        parametros.append(kwargs.copy())
        return real(**{**kwargs, "n_estimators": 8, "n_jobs": 1, "verbosity": -1})

    monkeypatch.setattr(modelado, "LGBMRegressor", constructor)
    return parametros


def test_validacion_temporal_hueco_y_disjuncion_con_test():
    partes, internas, fit, val = tablas_de_ejemplo()
    assert (fit["semana"] + pd.Timedelta(weeks=4) < internas["inicio_val"]).all()
    assert val["semana"].max() == partes["train"]["semana"].max()
    assert val["semana"].nunique() == 8
    assert val["semana"].min() == internas["inicio_val"]
    assert fit["semana"].max() == internas["inicio_val"] - pd.Timedelta(weeks=5)
    for tabla in (fit, val):
        assert tabla.index.intersection(partes["test"].index).empty
        assert tabla.index.isin(partes["train"].index).all()


def test_features_son_lista_explicita_sin_etiquetas_ni_futuro():
    _, _, fit, _ = tablas_de_ejemplo()
    fit["target_999"] = 123
    fit["columna_nueva"] = 456
    features = columnas_features(fit)
    assert len(features) == 17
    assert features[-1] == "segmento"
    assert not set(features) & {
        "repuesto", "semana", "grupo_historia", "entra_modelo", "frac_venta",
        "semanas_historia", "semanas_con_venta", "columna_nueva",
    }
    assert not any(c.startswith("target_") for c in features)
    with pytest.raises(ValueError, match="Faltan features"):
        columnas_features(fit.drop(columns="lag_1"))


@pytest.mark.parametrize("objetivo", ["l1", "tweedie"])
def test_entrena_cuatro_modelos_y_predice_sin_negativos(objetivo, modelos_chicos):
    _, _, fit, val = tablas_de_ejemplo()
    original_fit, original_val = fit.copy(deep=True), val.copy(deep=True)
    features = columnas_features(fit)
    modelos = entrenar_modelos(fit, val, features, objetivo)
    predicciones = predecir(modelos, val, features)
    assert set(modelos) == {1, 2, 3, 4}
    assert predicciones.columns.tolist() == ["pred_1", "pred_2", "pred_3", "pred_4"]
    assert predicciones.index.equals(val.index)
    assert np.isfinite(predicciones.to_numpy()).all()
    assert (predicciones >= 0).all().all()
    for modelo, parametros in zip(modelos.values(), modelos_chicos):
        assert parametros["n_estimators"] == 2000
        assert parametros["learning_rate"] == 0.05
        assert parametros["metric"] == "l1"
        assert parametros["random_state"] == 42
        assert modelo.booster_.pandas_categorical == [["frecuencia_alta", "intermitente"]]
        assert list(modelo.best_score_) == ["valid_0"]
        assert list(modelo.best_score_["valid_0"]) == ["l1"]
        assert modelo.best_iteration_ >= 1
    if objetivo == "tweedie":
        assert all(p["tweedie_variance_power"] == 1.5 for p in modelos_chicos)
    assert_frame_equal(fit, original_fit)
    assert_frame_equal(val, original_val)


def test_misma_semilla_da_mismas_predicciones(modelos_chicos):
    _, _, fit, val = tablas_de_ejemplo()
    features = columnas_features(fit)
    primero = entrenar_modelos(fit, val, features, "l1", semilla=7)
    segundo = entrenar_modelos(fit, val, features, "l1", semilla=7)
    assert_frame_equal(predecir(primero, val, features), predecir(segundo, val, features))


def tabla_manual():
    return pd.DataFrame({
        "repuesto": ["A", "A", "B", "B", "FUERA"],
        "entra_modelo": [True, True, True, True, False],
        "segmento": ["intermitente"] * 2 + ["frecuencia_alta"] * 3,
        "grupo_historia": ["10-25"] * 2 + ["26+"] * 2 + [pd.NA],
        "prom_8": [1, 1, 3, 3, 999],
        "target_1": [0, 2, 2, 4, 999],
        "target_2": [2, 0, 4, 2, 999],
    }, index=[10, 20, 30, 40, 50])


def test_evaluacion_a_mano_mismo_formato_y_filas_que_baselines():
    test = tabla_manual()
    original = test.copy(deep=True)
    pred = pd.DataFrame({"pred_1": [1, 1, 3, 3, 0], "pred_2": [1, 1, 3, 3, 0]}, index=test.index)
    resultado = evaluar_modelo(test, pred.sample(frac=1, random_state=2), "modelo")
    esperado = evaluar_baselines(test, horizonte=2)
    esperado = esperado.loc[esperado["baseline"] == "prom_8"].reset_index(drop=True)
    esperado["baseline"] = "modelo"
    assert_frame_equal(resultado, esperado)
    assert (resultado["mae"] == 1).all()
    assert (resultado.loc[resultado["agrupacion"] == "total", "n_filas"] == 4).all()
    assert_frame_equal(test, original)


def test_no_pierde_filas_si_promedio_falta_y_rechaza_predicciones_ausentes():
    test = tabla_manual()
    test.loc[10, "prom_8"] = np.nan
    pred = pd.DataFrame({"pred_1": [1] * 5, "pred_2": [1] * 5}, index=test.index)
    resultado = evaluar_modelo(test, pred, "modelo")
    assert (resultado.loc[resultado["agrupacion"] == "total", "n_filas"] == 4).all()
    with pytest.raises(ValueError, match="mismas filas"):
        evaluar_modelo(test, pred.drop(index=10), "modelo")
    pred.loc[10, "pred_1"] = np.nan
    with pytest.raises(ValueError, match="finitas"):
        evaluar_modelo(test, pred, "modelo")


def test_predecir_recorta_negativos_y_conserva_orden():
    class ModeloEjemplo:
        feature_name_ = ["cantidad", "segmento"]

        def predict(self, x):
            return np.array([-2.0, 3.0])

    df = pd.DataFrame({"cantidad": [0, 1], "segmento": ["intermitente", "frecuencia_alta"]}, index=[9, 3])
    pred = predecir({1: ModeloEjemplo()}, df, ModeloEjemplo.feature_name_)
    assert pred["pred_1"].tolist() == [0, 3]
    assert pred.index.tolist() == [9, 3]


def test_fuera_del_universo_no_participa_del_entrenamiento(modelos_chicos):
    _, _, fit, val = tablas_de_ejemplo()
    features = columnas_features(fit)
    modelos = entrenar_modelos(fit, val, features, "l1")
    excluida = fit.iloc[[0]].copy()
    excluida["entra_modelo"] = False
    excluida["cantidad"] = 99999
    excluida["segmento"] = "desconocido"
    otros = entrenar_modelos(pd.concat([fit, excluida]), val, features, "l1")
    assert_frame_equal(predecir(modelos, val, features), predecir(otros, val, features))


def test_etiquetas_internas_no_anticipan_validacion_ni_test():
    serie = serie_de_ejemplo()
    partes = dividir_por_fecha(construir_abt(serie))
    corte = separar_validacion(partes["train"])["inicio_val"]
    etiquetas = etiquetar_con_train(serie, corte)
    modificada = serie.copy()
    modificada.loc[modificada["semana"] >= corte, "cantidad"] = 1000
    assert_frame_equal(etiquetas, etiquetar_con_train(modificada, corte))


def test_parametros_y_fronteras_invalidas(modelos_chicos):
    _, _, fit, val = tablas_de_ejemplo()
    features = columnas_features(fit)
    with pytest.raises(ValueError, match="semanas_val"):
        separar_validacion(fit, semanas_val=0)
    with pytest.raises(ValueError, match="vacío"):
        separar_validacion(fit.iloc[:0])
    with pytest.raises(ValueError, match="objetivo"):
        entrenar_modelos(fit, val, features, "otro")
    with pytest.raises(ValueError, match="targets de fit"):
        entrenar_modelos(val, val, features, "l1")
    with pytest.raises(ValueError, match="permitidas"):
        entrenar_modelos(fit, val, features + ["target_1"], "l1")
    cero = fit.copy()
    cero[[f"target_{h}" for h in range(1, 5)]] = 0
    with pytest.raises(ValueError, match="Tweedie"):
        entrenar_modelos(cero, val, features, "tweedie")
