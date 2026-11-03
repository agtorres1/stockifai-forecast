import pandas as pd

from stockifai_forecast.datos import etiquetar_repuestos


def dividir_por_fecha(
    abt: pd.DataFrame, semanas_test: int = 26, horizonte: int = 4
) -> dict:
    """Separa por calendario y purga los orígenes cuyos targets tocan test.

    Espera semanas identificadas por lunes, como construir_abt. Las últimas
    semanas sin target completo quedan fuera de test, pero no cambian el corte.
    No modifica la ABT ni elimina esas filas del objeto original.
    """
    if not isinstance(semanas_test, int) or semanas_test < 1:
        raise ValueError("semanas_test debe ser un entero positivo")
    if not isinstance(horizonte, int) or horizonte < 1:
        raise ValueError("horizonte debe ser un entero positivo")
    if abt.empty:
        raise ValueError("la ABT no puede estar vacía para definir inicio_test")

    inicio_test = abt["semana"].max() - pd.Timedelta(weeks=semanas_test - 1)
    fin_targets = abt["semana"] + pd.Timedelta(weeks=horizonte)
    train = abt.loc[fin_targets < inicio_test].copy()
    test = abt.loc[
        (abt["semana"] >= inicio_test) & abt[f"target_{horizonte}"].notna()
    ].copy()
    return {"train": train, "test": test, "inicio_test": inicio_test}


def etiquetar_con_train(
    serie: pd.DataFrame,
    inicio_test: pd.Timestamp,
    min_semanas_con_venta: int = 10,
    umbral_frecuencia: float = 0.25,
) -> pd.DataFrame:
    """Etiqueta con toda la historia anterior al test, incluido el hueco.

    Los repuestos nuevos de test no aparecen en el resultado. Al unir con left
    merge quedan sin etiqueta y no entran al universo. grupo_historia es nulo
    fuera del universo; sus límites 10/26 son fijos, aun si cambia el mínimo.
    """
    historia = serie.loc[serie["semana"] < inicio_test]
    etiquetas = etiquetar_repuestos(
        historia,
        min_semanas_con_venta=min_semanas_con_venta,
        umbral_frecuencia=umbral_frecuencia,
    )
    etiquetas["grupo_historia"] = pd.Series(
        pd.NA, index=etiquetas.index, dtype="string"
    )
    elegibles = etiquetas["entra_modelo"]
    ventas = etiquetas["semanas_con_venta"]
    etiquetas.loc[elegibles & ventas.between(10, 25), "grupo_historia"] = "10-25"
    etiquetas.loc[elegibles & ventas.ge(26), "grupo_historia"] = "26+"
    return etiquetas


def evaluar_baselines(
    test_etiquetado: pd.DataFrame, horizonte: int = 4
) -> pd.DataFrame:
    """MAE por fila SKU-semana, separado por segmento, historia y total.

    Salida larga: baseline, horizonte, agrupacion, grupo, mae, n_candidatas,
    n_filas, n_excluidas, n_sin_target. n_excluidas cuenta predicciones faltantes
    entre filas con target conocido (prom_8 NaN). n_sin_target se informa aparte.
    Por celda: n_candidatas = n_filas + n_excluidas + n_sin_target.
    Una celda sin observaciones evaluables tiene mae NaN, no cero.
    """
    if not isinstance(horizonte, int) or horizonte < 1:
        raise ValueError("horizonte debe ser un entero positivo")
    universo = test_etiquetado.loc[
        test_etiquetado["entra_modelo"].eq(True).fillna(False)
    ].copy()
    grupos = [("total", "total", universo)]
    for dimension in ("segmento", "grupo_historia"):
        grupos.extend(
            (dimension, nombre, filas)
            for nombre, filas in universo.groupby(
                dimension, sort=True, dropna=False, observed=True
            )
        )

    resultados = []
    for h in range(1, horizonte + 1):
        for dimension, nombre, filas in grupos:
            target = filas[f"target_{h}"]
            for baseline in ("cero", "prom_8"):
                prediccion = (
                    pd.Series(0.0, index=filas.index)
                    if baseline == "cero" else filas["prom_8"]
                )
                conocidas = target.notna()
                usadas = conocidas & prediccion.notna()
                error = (target.loc[usadas] - prediccion.loc[usadas]).abs()
                resultados.append({
                    "baseline": baseline,
                    "horizonte": h,
                    "agrupacion": dimension,
                    "grupo": nombre,
                    "mae": error.mean(),
                    "n_candidatas": len(filas),
                    "n_filas": int(usadas.sum()),
                    "n_excluidas": int((conocidas & prediccion.isna()).sum()),
                    "n_sin_target": int((~conocidas).sum()),
                })
    return pd.DataFrame(resultados)
