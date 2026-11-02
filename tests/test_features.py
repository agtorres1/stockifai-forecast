import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from stockifai_forecast.features import construir_abt


def serie_de_ejemplo():
    semanas = pd.date_range("2025-06-02", periods=16, freq="W-MON")
    return pd.DataFrame({
        "repuesto": ["A"] * 16 + ["B"] * 16,
        "semana": list(semanas) * 2,
        "cantidad": [2, 0, 4, 0, 0, 6, 0, 0, 0, 3, 0, 0, 5, 0, 0, 1]
        + [10, 0, 0, 20, 0, 0, 0, 30, 0, 0, 0, 0, 40, 0, 0, 0],
    })


def test_frontera_no_mezcla_repuestos():
    abt = construir_abt(serie_de_ejemplo())
    primera_b = abt[abt["repuesto"] == "B"].iloc[0]
    ultima_a = abt[abt["repuesto"] == "A"].iloc[-1]

    assert pd.isna(primera_b["lag_1"])
    assert pd.isna(primera_b["prom_4"])
    assert pd.isna(ultima_a["target_1"])


def test_features_no_cambian_si_cambian_ventas_futuras():
    serie = serie_de_ejemplo()
    corte = pd.Timestamp("2025-07-21")
    original = construir_abt(serie)
    modificada = serie.copy()
    modificada.loc[modificada["semana"] > corte, "cantidad"] = 999
    nueva = construir_abt(modificada)
    features = [c for c in original.columns if not c.startswith("target_")]
    filas_corte = original["semana"] == corte

    assert_frame_equal(
        original.loc[filas_corte, features], nueva.loc[filas_corte, features]
    )
    # El futuro sí cambió: el test compara features, no una copia sin alterar.
    assert (nueva.loc[filas_corte, "target_1"] == 999).all()
    assert (original.loc[filas_corte, "target_1"] != 999).all()


def test_targets_corresponden_al_mismo_repuesto_y_horizonte():
    serie = serie_de_ejemplo()
    abt = construir_abt(serie)
    cantidades = serie.set_index(["repuesto", "semana"])["cantidad"]

    for _, fila in abt.iterrows():
        for h in range(1, 5):
            clave = (fila["repuesto"], fila["semana"] + pd.Timedelta(weeks=h))
            esperado = cantidades.get(clave, float("nan"))
            if pd.isna(esperado):
                assert pd.isna(fila[f"target_{h}"])
            else:
                assert fila[f"target_{h}"] == esperado


def test_features_coinciden_con_valores_a_mano():
    abt = construir_abt(serie_de_ejemplo())
    a = abt[abt["repuesto"] == "A"].reset_index(drop=True)
    b = abt[abt["repuesto"] == "B"].reset_index(drop=True)

    assert a.loc[3, "prom_4"] == 1.5  # (2 + 0 + 4 + 0) / 4
    assert a.loc[7, "prom_4"] == 1.5  # (0 + 6 + 0 + 0) / 4
    assert a.loc[7, "semanas_con_venta_8"] == 3
    assert a.loc[7, "semanas_desde_ultima_venta"] == 2
    assert a.loc[9, "semanas_desde_ultima_venta"] == 0
    assert a.loc[7, "desvio_8"] == pytest.approx((38 / 7) ** 0.5)
    assert a.loc[9, "cantidad"] == 3
    assert a.loc[9, ["lag_1", "lag_2", "lag_3"]].tolist() == [0, 0, 0]
    assert b.loc[7, "semanas_con_venta_8"] == 3
    assert b.loc[7, "semanas_desde_ultima_venta"] == 0


def test_nan_en_ventanas_incompletas_y_targets_sin_futuro():
    abt = construir_abt(serie_de_ejemplo())
    for _, grupo in abt.groupby("repuesto"):
        for n in (4, 8, 12):
            assert grupo[f"prom_{n}"].iloc[:n - 1].isna().all()
            assert grupo[f"prom_{n}"].iloc[n - 1:].notna().all()
        for columna in ("desvio_8", "semanas_con_venta_8"):
            assert grupo[columna].iloc[:7].isna().all()
            assert grupo[columna].iloc[7:].notna().all()
        for h in range(1, 5):
            assert grupo[f"target_{h}"].iloc[-h:].isna().all()
            assert grupo[f"target_{h}"].iloc[:-h].notna().all()


def test_conserva_filas_claves_y_entrada():
    serie = serie_de_ejemplo()
    original = serie.copy(deep=True)
    abt = construir_abt(serie)

    assert len(abt) == len(serie)
    assert not abt.duplicated(["repuesto", "semana"]).any()
    assert_frame_equal(abt[["repuesto", "semana", "cantidad"]], original)
    assert_frame_equal(serie, original)


def test_resultado_independiente_del_orden_de_entrada():
    serie = serie_de_ejemplo()
    mezclada = serie.sample(frac=1, random_state=42)

    assert_frame_equal(construir_abt(serie), construir_abt(mezclada))


def test_feriado_del_9_de_julio_en_la_semana_siguiente():
    abt = construir_abt(serie_de_ejemplo())
    anterior = abt[abt["semana"] == pd.Timestamp("2025-06-30")]

    assert len(anterior) == 2
    assert (anterior["feriados_t1"] == 1).all()
    # Desde el 23/06, esa misma semana futura está a dos semanas.
    dos_antes = abt[abt["semana"] == pd.Timestamp("2025-06-23")]
    assert (dos_antes["feriados_t2"] == 1).all()


def test_calendario_iso_y_feriados_al_cruzar_el_anio():
    serie = pd.DataFrame({
        "repuesto": ["A", "B"],
        "semana": pd.to_datetime(["2025-12-22", "2025-12-29"]),
        "cantidad": [1, 2],
    })
    abt = construir_abt(serie)

    assert abt["feriados_t1"].tolist() == [1, 0]  # 1 de enero; luego ninguno
    assert abt["mes"].tolist() == [12, 12]
    assert abt["semana_del_anio"].tolist() == [52, 1]
    assert abt["target_1"].isna().all()  # calendario conocido, ventas desconocidas


def test_ventanas_y_horizonte_personalizados():
    abt = construir_abt(serie_de_ejemplo(), ventanas=(2, 5), horizonte=6)

    assert "prom_4" not in abt.columns
    assert abt.loc[1, "prom_2"] == 1
    assert pd.isna(abt.loc[3, "prom_5"])
    assert abt.loc[4, "prom_5"] == 1.2
    assert abt.loc[0, "target_6"] == 0
    assert "feriados_t6" in abt.columns
    assert "target_7" not in abt.columns


def test_historia_sin_venta_previa_no_inventa_antiguedad():
    serie = pd.DataFrame({
        "repuesto": ["A"] * 3 + ["B"] * 3,
        "semana": list(pd.date_range("2025-06-02", periods=3, freq="W-MON")) * 2,
        "cantidad": [0, 0, 2, 0, 0, 0],
    })
    abt = construir_abt(serie)

    assert abt.loc[:1, "semanas_desde_ultima_venta"].isna().all()
    assert abt.loc[2, "semanas_desde_ultima_venta"] == 0
    assert abt.loc[3:, "semanas_desde_ultima_venta"].isna().all()


def test_entrada_vacia_conserva_esquema():
    serie = serie_de_ejemplo().iloc[:0]
    abt = construir_abt(serie)

    assert abt.empty
    assert "prom_12" in abt.columns
    assert "feriados_t4" in abt.columns
    assert "target_4" in abt.columns


def test_parametros_invalidos():
    with pytest.raises(ValueError, match="horizonte"):
        construir_abt(serie_de_ejemplo(), horizonte=0)
    with pytest.raises(ValueError, match="ventanas"):
        construir_abt(serie_de_ejemplo(), ventanas=(0,))
