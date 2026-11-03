import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from stockifai_forecast.evaluacion import (
    dividir_por_fecha,
    etiquetar_con_train,
    evaluar_baselines,
)
from stockifai_forecast.features import construir_abt


def serie_de_ejemplo():
    semanas = pd.date_range("2024-01-01", periods=60, freq="W-MON")
    return pd.DataFrame({
        "repuesto": ["A"] * 60 + ["B"] * 60,
        "semana": list(semanas) * 2,
        "cantidad": [1] * 60 + [2 if i % 3 == 0 else 0 for i in range(60)],
    })


def test_train_no_tiene_targets_en_test():
    abt = construir_abt(serie_de_ejemplo())
    partes = dividir_por_fecha(abt)

    assert partes["inicio_test"] == pd.Timestamp("2024-08-26")
    assert partes["inicio_test"].dayofweek == 0
    assert len(partes["train"]) == 60  # 30 semanas por SKU
    for h in range(1, 5):
        assert (
            partes["train"]["semana"] + pd.Timedelta(weeks=h)
            < partes["inicio_test"]
        ).all()


def test_hueco_de_cuatro_semanas_no_aparece_en_train_ni_test():
    abt = construir_abt(serie_de_ejemplo())
    partes = dividir_por_fecha(abt)
    corte = partes["inicio_test"]
    hueco = abt.loc[
        (abt["semana"] >= corte - pd.Timedelta(weeks=4))
        & (abt["semana"] < corte)
    ]

    assert len(hueco) == 8
    for nombre in ("train", "test"):
        assert hueco.index.intersection(partes[nombre].index).empty
    # La fila anterior al hueco sí entra, la que toca exactamente el corte no.
    assert partes["train"]["semana"].max() == corte - pd.Timedelta(weeks=5)


def test_test_tiene_futuro_completo_sin_mover_el_corte():
    abt = construir_abt(serie_de_ejemplo())
    original = abt.copy(deep=True)
    partes = dividir_por_fecha(abt)

    assert len(partes["test"]) == 44  # 22 orígenes por SKU
    assert partes["test"]["target_4"].notna().all()
    assert partes["test"]["semana"].min() == partes["inicio_test"]
    assert partes["test"]["semana"].max() == abt["semana"].max() - pd.Timedelta(weeks=4)
    assert_frame_equal(abt, original)


def test_etiquetas_no_dependen_de_ventas_del_test():
    serie = serie_de_ejemplo()
    inicio = pd.Timestamp("2024-08-26")
    etiquetas = etiquetar_con_train(serie, inicio)
    modificada = serie.copy()
    modificada.loc[modificada["semana"] >= inicio, "cantidad"] = 999

    assert_frame_equal(etiquetas, etiquetar_con_train(modificada, inicio))
    a, b = etiquetas.set_index("repuesto").loc[["A", "B"]].to_dict("records")
    assert a["semanas_con_venta"] == 34
    assert a["grupo_historia"] == "26+"
    assert b["semanas_con_venta"] == 12
    assert b["grupo_historia"] == "10-25"
    # Las cuatro semanas del hueco sí cuentan como historia conocida.
    assert a["semanas_historia"] == 34


def test_limites_de_historia_universo_y_repuesto_nuevo():
    semanas = pd.date_range("2024-01-01", periods=40, freq="W-MON")
    serie = pd.concat([
        pd.DataFrame({"repuesto": sku, "semana": semanas,
                      "cantidad": [1] * n + [0] * (40 - n)})
        for sku, n in [("A", 9), ("B", 10), ("C", 25), ("D", 26)]
    ], ignore_index=True)
    inicio = semanas[-1] + pd.Timedelta(weeks=1)
    nuevo = pd.DataFrame({"repuesto": ["NUEVO"], "semana": [inicio], "cantidad": [99]})
    etiquetas = etiquetar_con_train(pd.concat([serie, nuevo]), inicio).set_index("repuesto")

    assert not etiquetas.loc["A", "entra_modelo"]
    assert pd.isna(etiquetas.loc["A", "grupo_historia"])
    assert etiquetas.loc["B", "grupo_historia"] == "10-25"
    assert etiquetas.loc["C", "grupo_historia"] == "10-25"
    assert etiquetas.loc["D", "grupo_historia"] == "26+"
    assert "NUEVO" not in etiquetas.index


def test_etiquetas_respetan_parametros_y_no_modifican_serie():
    serie = serie_de_ejemplo()
    original = serie.copy(deep=True)
    etiquetas = etiquetar_con_train(
        serie, pd.Timestamp("2024-08-26"),
        min_semanas_con_venta=20, umbral_frecuencia=0.5,
    ).set_index("repuesto")

    assert not etiquetas.loc["B", "entra_modelo"]
    assert etiquetas.loc["B", "segmento"] == "intermitente"
    assert pd.isna(etiquetas.loc["B", "grupo_historia"])
    assert_frame_equal(serie, original)


def test_etiquetado_de_historia_vacia():
    serie = serie_de_ejemplo()
    etiquetas = etiquetar_con_train(serie, serie["semana"].min())

    assert etiquetas.empty
    assert "grupo_historia" in etiquetas.columns


def tabla_de_ejemplo():
    return pd.DataFrame({
        "repuesto": ["A", "A", "B", "B", "FUERA"],
        "entra_modelo": [True, True, True, True, False],
        "segmento": ["intermitente"] * 2 + ["frecuencia_alta"] * 3,
        "grupo_historia": ["10-25"] * 2 + ["26+"] * 2 + [pd.NA],
        "prom_8": [1.0, 1.0, 3.0, 3.0, 1000.0],
        "target_1": [0.0, 2.0, 2.0, 4.0, 1000.0],
        "target_2": [2.0, 0.0, 4.0, 2.0, 1000.0],
    })


def celda(resultado, baseline, agrupacion="total", grupo="total", h=1):
    return resultado.loc[
        (resultado["baseline"] == baseline)
        & (resultado["horizonte"] == h)
        & (resultado["agrupacion"] == agrupacion)
        & (resultado["grupo"] == grupo)
    ].iloc[0]


def test_mae_coincide_con_calculo_manual_por_dimension_y_horizonte():
    resultado = evaluar_baselines(tabla_de_ejemplo(), horizonte=2)

    assert len(resultado) == 20  # 2 reglas x 2 horizontes x (total + 2 + 2)
    for h in (1, 2):
        assert celda(resultado, "cero", h=h)["mae"] == 2.0
        assert celda(resultado, "prom_8", h=h)["mae"] == 1.0
        assert celda(resultado, "cero", "segmento", "intermitente", h)["mae"] == 1.0
        assert celda(resultado, "cero", "segmento", "frecuencia_alta", h)["mae"] == 3.0
        assert celda(resultado, "prom_8", "grupo_historia", "10-25", h)["mae"] == 1.0
        assert celda(resultado, "cero", "grupo_historia", "26+", h)["mae"] == 3.0
        assert celda(resultado, "cero", h=h)["n_filas"] == 4


def test_fuera_del_universo_no_afecta_resultados():
    test = tabla_de_ejemplo()
    original = test.copy(deep=True)
    resultado = evaluar_baselines(test, horizonte=2)
    solo_universo = evaluar_baselines(test.loc[test["entra_modelo"]], horizonte=2)

    assert_frame_equal(resultado, solo_universo)
    assert_frame_equal(test, original)


def test_nan_de_promedio_excluye_solo_media_movil_y_se_cuenta():
    test = tabla_de_ejemplo()
    test.loc[0, "prom_8"] = float("nan")
    resultado = evaluar_baselines(test, horizonte=2)

    for h in (1, 2):
        cero = celda(resultado, "cero", h=h)
        promedio = celda(resultado, "prom_8", h=h)
        assert cero["n_filas"] == 4
        assert cero["n_excluidas"] == 0
        assert promedio["n_filas"] == 3
        assert promedio["n_excluidas"] == 1
        assert celda(resultado, "prom_8", "segmento", "intermitente", h)["n_excluidas"] == 1
        assert celda(resultado, "prom_8", "segmento", "frecuencia_alta", h)["n_excluidas"] == 0


def test_sin_target_se_informa_aparte_y_celda_sin_datos_no_es_cero():
    test = tabla_de_ejemplo()
    test.loc[:1, "prom_8"] = float("nan")
    test.loc[2, "target_1"] = float("nan")
    resultado = evaluar_baselines(test, horizonte=2)

    assert pd.isna(celda(resultado, "prom_8", "segmento", "intermitente")["mae"])
    assert celda(resultado, "cero")["n_sin_target"] == 1
    assert celda(resultado, "cero", h=2)["n_sin_target"] == 0
    assert (
        resultado["n_candidatas"] == resultado["n_filas"]
        + resultado["n_excluidas"] + resultado["n_sin_target"]
    ).all()


def test_universo_vacio_y_etiquetas_faltantes():
    test = tabla_de_ejemplo()
    test["entra_modelo"] = pd.NA
    resultado = evaluar_baselines(test, horizonte=2)

    assert len(resultado) == 4  # solo totales de cada regla y horizonte
    assert resultado["mae"].isna().all()
    assert (resultado["n_filas"] == 0).all()


def test_split_con_otro_horizonte_y_orden_mezclado():
    abt = construir_abt(serie_de_ejemplo(), horizonte=2)
    esperado = dividir_por_fecha(abt, semanas_test=8, horizonte=2)
    mezclado = dividir_por_fecha(abt.sample(frac=1, random_state=3), semanas_test=8, horizonte=2)

    assert mezclado["inicio_test"] == esperado["inicio_test"]
    assert set(mezclado["train"].index) == set(esperado["train"].index)
    assert set(mezclado["test"].index) == set(esperado["test"].index)


def test_split_vacio_y_parametros_invalidos():
    abt = construir_abt(serie_de_ejemplo())
    with pytest.raises(ValueError, match="vacía"):
        dividir_por_fecha(abt.iloc[:0])
    with pytest.raises(ValueError, match="semanas_test"):
        dividir_por_fecha(abt, semanas_test=0)
    with pytest.raises(ValueError, match="horizonte"):
        dividir_por_fecha(abt, horizonte=0)
    with pytest.raises(ValueError, match="horizonte"):
        evaluar_baselines(tabla_de_ejemplo(), horizonte=0)
