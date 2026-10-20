import pandas as pd

from stockifai_forecast.datos import (
    agregar_semanal,
    etiquetar_repuestos,
    leer_movimientos,
    rellenar_ceros,
)


# ---------- leer_movimientos ----------

def test_leer_movimientos_filtra_egresos_y_renombra(tmp_path):
    excel = pd.DataFrame({
        "KardexId": [1, 2, 3],
        "Repuesto": ["A", "A", "B"],
        "Fecha": pd.to_datetime(["2025-07-07", "2025-07-08", "2025-07-09"]),
        "Cantidad": [1, 4, 2],
        "Tipo": ["E", "I", "E"],
        "Deposito": ["CENTRAL", "CENTRAL", "D3"],
    })
    ruta = tmp_path / "movimientos.xlsx"
    excel.to_excel(ruta, index=False)

    mov = leer_movimientos(ruta)

    assert list(mov.columns) == ["repuesto", "fecha", "cantidad"]
    assert len(mov) == 2
    assert mov["cantidad"].tolist() == [1, 2]


# ---------- agregar_semanal ----------

def test_agregar_semanal_suma_dentro_de_la_semana():
    mov = pd.DataFrame({
        "repuesto": ["A", "A", "A"],
        "fecha": pd.to_datetime(["2025-07-07", "2025-07-09", "2025-07-21"]),
        "cantidad": [1, 2, 5],
    })

    semanal = agregar_semanal(mov)

    assert len(semanal) == 2
    assert semanal["cantidad"].tolist() == [3, 5]


def test_agregar_semanal_usa_el_lunes_como_semana():
    mov = pd.DataFrame({
        "repuesto": ["A"],
        "fecha": pd.to_datetime(["2025-07-09"]),
        "cantidad": [1],
    })

    semanal = agregar_semanal(mov)

    assert semanal["semana"].iloc[0] == pd.Timestamp("2025-07-07")


# ---------- rellenar_ceros ----------

def semanal_de_ejemplo():
    return pd.DataFrame({
        "repuesto": ["A", "A", "B"],
        "semana": pd.to_datetime(["2025-07-07", "2025-07-21", "2025-07-14"]),
        "cantidad": [3, 5, 2],
    })


def test_rellenar_ceros_completa_los_huecos():
    serie = rellenar_ceros(semanal_de_ejemplo())
    serie_a = serie[serie["repuesto"] == "A"]

    assert serie_a["cantidad"].tolist() == [3, 0, 5]


def test_rellenar_ceros_arranca_en_la_primera_venta():
    serie = rellenar_ceros(semanal_de_ejemplo())
    serie_b = serie[serie["repuesto"] == "B"]

    assert serie_b["semana"].min() == pd.Timestamp("2025-07-14")
    assert serie_b["cantidad"].tolist() == [2, 0]


def test_rellenar_ceros_llega_hasta_fin():
    fin = pd.Timestamp("2025-08-04")

    serie = rellenar_ceros(semanal_de_ejemplo(), fin=fin)

    assert (serie.groupby("repuesto")["semana"].max() == fin).all()


def test_rellenar_ceros_no_cambia_el_total():
    semanal = semanal_de_ejemplo()

    serie = rellenar_ceros(semanal)

    assert serie["cantidad"].sum() == semanal["cantidad"].sum()


def test_rellenar_ceros_no_duplica_filas():
    serie = rellenar_ceros(semanal_de_ejemplo())

    assert not serie.duplicated(subset=["repuesto", "semana"]).any()


# ---------- etiquetar_repuestos ----------

def test_etiquetar_repuestos_cuenta_y_segmenta():
    serie = pd.DataFrame({
        "repuesto": ["A"] * 4 + ["B"] * 4,
        "semana": pd.to_datetime(["2025-07-07", "2025-07-14", "2025-07-21", "2025-07-28"] * 2),
        "cantidad": [1, 2, 0, 3,   0, 0, 0, 1],
    })

    resumen = etiquetar_repuestos(serie, min_semanas_con_venta=2, umbral_frecuencia=0.5)
    a = resumen[resumen["repuesto"] == "A"].iloc[0]
    b = resumen[resumen["repuesto"] == "B"].iloc[0]

    assert a["semanas_historia"] == 4
    assert a["semanas_con_venta"] == 3
    assert a["entra_modelo"]
    assert a["segmento"] == "frecuencia_alta"

    assert b["semanas_con_venta"] == 1
    assert not b["entra_modelo"]
    assert b["segmento"] == "intermitente"