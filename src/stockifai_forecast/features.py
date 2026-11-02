import holidays
import pandas as pd


def construir_abt(
    serie: pd.DataFrame,
    ventanas: tuple[int, ...] = (4, 8, 12),
    horizonte: int = 4,
) -> pd.DataFrame:
    """Construye features al cierre de t y targets de t+1 a t+horizonte.

    La entrada es la serie semanal completa de rellenar_ceros: una fila por
    repuesto y lunes, sin huecos ni cantidades faltantes. No modifica la entrada
    ni elimina filas. Las ventanas incluyen t; desvio_8 usa ddof=1 (muestral).
    Antes de la primera venta, semanas_desde_ultima_venta queda NaN.
    """
    if not isinstance(horizonte, int) or horizonte < 1:
        raise ValueError("horizonte debe ser un entero positivo")
    ventanas = tuple(ventanas)
    if any(not isinstance(n, int) or n < 1 for n in ventanas):
        raise ValueError("las ventanas deben ser enteros positivos")

    # Copia ordenada: los desplazamientos siguen el tiempo dentro de cada SKU.
    abt = serie[["repuesto", "semana", "cantidad"]].copy()
    abt = abt.sort_values(["repuesto", "semana"]).reset_index(drop=True)
    grupos = abt.groupby("repuesto", sort=False)
    cantidades = grupos["cantidad"]

    for lag in (1, 2, 3):
        abt[f"lag_{lag}"] = cantidades.shift(lag)

    for n in ventanas:
        abt[f"prom_{n}"] = cantidades.transform(
            lambda x: x.rolling(n, min_periods=n).mean()
        )
    abt["desvio_8"] = cantidades.transform(
        lambda x: x.rolling(8, min_periods=8).std(ddof=1)
    )
    abt["semanas_con_venta_8"] = cantidades.transform(
        lambda x: x.gt(0).rolling(8, min_periods=8).sum()
    )

    # Conserva la fecha de la última venta conocida, sin mirar hacia adelante.
    ultima_venta = abt["semana"].where(abt["cantidad"].gt(0))
    ultima_venta = ultima_venta.groupby(abt["repuesto"], sort=False).ffill()
    abt["semanas_desde_ultima_venta"] = (
        abt["semana"] - ultima_venta
    ).dt.days / 7

    abt["mes"] = abt["semana"].dt.month
    abt["semana_del_anio"] = abt["semana"].dt.isocalendar().week

    # Calendario nacional (sin subdivisión provincial), de lunes a domingo.
    # Cuenta una sola vez cada semana distinta, aunque la compartan muchos SKUs.
    semanas = abt["semana"].drop_duplicates()
    if abt.empty:
        calendario = holidays.Argentina(years=[])
    else:
        primer_dia = semanas.min() + pd.Timedelta(weeks=1)
        ultimo_dia = semanas.max() + pd.Timedelta(weeks=horizonte, days=6)
        calendario = holidays.Argentina(
            years=range(primer_dia.year, ultimo_dia.year + 1)
        )
    conteos = {}
    for h in range(1, horizonte + 1):
        futuras = semanas + pd.Timedelta(weeks=h)
        for lunes in futuras:
            if lunes not in conteos:
                conteos[lunes] = sum(
                    (lunes + pd.Timedelta(days=d)).date() in calendario
                    for d in range(7)
                )
        abt[f"feriados_t{h}"] = (
            abt["semana"] + pd.Timedelta(weeks=h)
        ).map(conteos).astype(int)

    for h in range(1, horizonte + 1):
        abt[f"target_{h}"] = cantidades.shift(-h)

    return abt
