import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import stockifai_forecast.modelado as modelado
from stockifai_forecast.backtesting import calcular_cortes, correr_examen, correr_backtesting, resumir
from stockifai_forecast.evaluacion import dividir_por_fecha, etiquetar_con_train, evaluar_baselines
from stockifai_forecast.features import construir_abt


@pytest.fixture
def datos():
    semanas = pd.date_range("2020-01-06", periods=180, freq="W-MON")
    serie = pd.DataFrame({
        "repuesto": ["A"] * 180 + ["B"] * 180,
        "semana": list(semanas) * 2,
        "cantidad": [i % 5 for i in range(180)]
        + [1 + i % 9 if i % 6 == 0 else 0 for i in range(180)],
    })
    return serie, construir_abt(serie)


@pytest.fixture(autouse=True)
def modelos_chicos(monkeypatch):
    real = modelado.LGBMRegressor

    def constructor(**kwargs):
        return real(**{**kwargs, "n_estimators": 8, "n_jobs": 1, "verbosity": -1})

    monkeypatch.setattr(modelado, "LGBMRegressor", constructor)


def test_cortes_separados_y_ultimo_igual_al_corte_original(datos):
    _, abt = datos
    cortes = calcular_cortes(abt)
    assert len(cortes) == 4
    assert all(c.dayofweek == 0 for c in cortes)
    assert cortes[-1] == dividir_por_fecha(abt)["inicio_test"]
    assert all(b - a == pd.Timedelta(weeks=26) for a, b in zip(cortes, cortes[1:]))


def test_ventanas_no_se_pisan(datos):
    _, abt = datos
    ventanas = [set(pd.date_range(c, periods=26, freq="W-MON")) for c in calcular_cortes(abt)]
    for i, izquierda in enumerate(ventanas):
        for derecha in ventanas[i + 1:]:
            assert izquierda.isdisjoint(derecha)


def test_cada_examen_respeta_train_fit_val_test_y_targets_de_ventana(datos):
    serie, abt = datos
    original_serie, original_abt = serie.copy(deep=True), abt.copy(deep=True)
    for inicio in calcular_cortes(abt):
        resultado = correr_examen(serie, abt, inicio)
        indices = resultado["indices"]
        train, fit, val, test = [abt.loc[indices[n]] for n in ("train", "fit", "val", "test")]
        assert (train["semana"] + pd.Timedelta(weeks=4) < inicio).all()
        assert (fit["semana"] + pd.Timedelta(weeks=4) < resultado["inicio_val"]).all()
        assert val["semana"].max() <= train["semana"].max()
        assert set(test.index).isdisjoint(fit.index)
        assert set(test.index).isdisjoint(val.index)
        assert set(test.index).isdisjoint(train.index)
        assert test["semana"].nunique() == 22
        for h in range(1, 5):
            fecha_target = test["semana"] + pd.Timedelta(weeks=h)
            assert (fecha_target >= inicio).all()
            assert (fecha_target < resultado["limite_test"]).all()
            assert test[f"target_{h}"].notna().all()
        assert resultado["filas"]["test"] == 44
    assert_frame_equal(serie, original_serie)
    assert_frame_equal(abt, original_abt)


def test_etiquetas_de_cada_examen_no_miran_ventas_posteriores(datos):
    serie, abt = datos
    for inicio in calcular_cortes(abt):
        resultado = correr_examen(serie, abt, inicio)
        modificada = serie.copy()
        modificada.loc[modificada["semana"] >= inicio, "cantidad"] = 999
        otro = correr_examen(modificada, construir_abt(modificada), inicio)
        assert_frame_equal(resultado["etiquetas_test"], otro["etiquetas_test"])
        assert_frame_equal(resultado["etiquetas_fit_val"], otro["etiquetas_fit_val"])
        assert resultado["repuestos"] == otro["repuestos"]
        assert resultado["mejores_iteraciones"] == otro["mejores_iteraciones"]


def test_tres_metodos_usan_las_mismas_filas_y_targets(datos):
    serie, abt = datos
    resultado = correr_examen(serie, abt, calcular_cortes(abt)[0])
    metricas = resultado["metricas"]
    assert set(metricas["baseline"]) == {"cero", "prom_8", "lightgbm_l1"}
    cantidades = metricas.pivot(index=["horizonte", "agrupacion", "grupo"],
                               columns="baseline", values="n_filas")
    assert (cantidades["cero"] == cantidades["prom_8"]).all()
    assert (cantidades["cero"] == cantidades["lightgbm_l1"]).all()
    etiquetas = resultado["etiquetas_test"].set_index("repuesto")
    test = abt.loc[resultado["indices"]["test"]].join(etiquetas, on="repuesto")
    esperadas = evaluar_baselines(test)
    assert_frame_equal(metricas.loc[metricas["baseline"] != "lightgbm_l1"].reset_index(drop=True), esperadas)


def test_ultimo_examen_reproduce_flujo_de_notebook_05(datos):
    serie, abt = datos
    partes = dividir_por_fecha(abt)
    internas = modelado.separar_validacion(partes["train"])
    etiquetas = etiquetar_con_train(serie, internas["inicio_val"]).set_index("repuesto")
    fit, val = [internas[n].join(etiquetas, on="repuesto") for n in ("fit", "val")]
    etiquetas_test = etiquetar_con_train(serie, partes["inicio_test"]).set_index("repuesto")
    test = partes["test"].join(etiquetas_test, on="repuesto")
    test = test.loc[test["entra_modelo"].eq(True)]
    features = modelado.columnas_features(fit)
    modelos = modelado.entrenar_modelos(fit, val, features, "l1")
    predicciones = modelado.predecir(modelos, test, features)
    esperado = modelado.evaluar_modelo(test, predicciones, "lightgbm_l1")
    resultado = correr_examen(serie, abt, partes["inicio_test"])["metricas"]
    observado = resultado.loc[resultado["baseline"] == "lightgbm_l1"].reset_index(drop=True)
    assert_frame_equal(observado, esperado)


def test_promedio_faltante_excluye_la_misma_clave_de_todos(datos):
    serie, abt = datos
    inicio = calcular_cortes(abt)[0]
    clave = abt.index[(abt["semana"] == inicio) & (abt["repuesto"] == "A")][0]
    abt = abt.copy()
    abt.loc[clave, "prom_8"] = np.nan
    resultado = correr_examen(serie, abt, inicio)
    assert resultado["filas"]["test_excluidas_sin_prom_8"] == 1
    assert clave not in resultado["indices"]["test"]
    totales = resultado["metricas"].loc[resultado["metricas"]["agrupacion"] == "total"]
    assert (totales["n_filas"] == 43).all()


def test_backtesting_devuelve_examenes_y_detalles_sin_tracking_por_defecto(datos):
    serie, abt = datos
    metricas = correr_backtesting(serie, abt, n_examenes=2)
    assert metricas["examen"].unique().tolist() == [1, 2]
    assert len(metricas.attrs["examenes"]) == 2
    assert metricas.attrs["run_ids"] == {}
    assert "indices" not in metricas.attrs["examenes"][0]
    assert metricas.groupby("examen")["inicio_test"].first().tolist() == calcular_cortes(abt, n_examenes=2)


def test_resumen_promedia_examenes_sin_ponderar_filas_y_cuenta_empates():
    filas = []
    for examen, valores in [(1, {"cero": 1, "prom_8": 2, "lightgbm_l1": 1}),
                            (2, {"cero": 5, "prom_8": 2, "lightgbm_l1": 3})]:
        for metodo, mae in valores.items():
            for h in range(1, 5):
                filas.append({"examen": examen, "baseline": metodo, "agrupacion": "total",
                              "grupo": "total", "horizonte": h, "mae": mae,
                              "n_filas": 10 if examen == 1 else 1000})
    tabla = resumir(pd.DataFrame(filas)).set_index("baseline")
    assert tabla.loc["cero", "mae_promedio"] == 3
    assert tabla.loc["cero", "mae_min"] == 1
    assert tabla.loc["cero", "mae_max"] == 5
    assert tabla["examenes_ganados"].to_dict() == {"cero": 1, "lightgbm_l1": 1, "prom_8": 1}
    assert (tabla["n_examenes"] == 2).all()


def test_cortes_y_ventanas_invalidas_fallan_con_mensaje(datos):
    serie, abt = datos
    with pytest.raises(ValueError, match="n_examenes"):
        calcular_cortes(abt, n_examenes=0)
    with pytest.raises(ValueError, match="superar"):
        calcular_cortes(abt, semanas_test=4, horizonte=4)
    with pytest.raises(ValueError, match="vacía"):
        calcular_cortes(abt.iloc[:0])
    with pytest.raises(ValueError, match="historia suficiente"):
        correr_examen(serie, abt, abt["semana"].min())
    with pytest.raises(ValueError, match="excede"):
        correr_examen(serie, abt, abt["semana"].max())
    assert resumir(pd.DataFrame()).empty
