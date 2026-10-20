
import pandas as pd


def leer_movimientos(ruta:str) -> pd.DataFrame:
    mov = pd.read_excel(ruta)
    mov = mov[mov['Tipo']=='E']
    mov = mov.rename(columns={'Repuesto': 'repuesto','Fecha':'fecha','Cantidad':'cantidad'})
    return mov[["repuesto", "fecha", "cantidad"]]


"""devuele la tabla semanal"""
"""repuesto semana cantidad"""
def agregar_semanal(mov: pd.DataFrame) -> pd.DataFrame: 
    """suma las cantdades por repuesto y semana (semanas que emppiezan en lunes) """ 
    mov = mov.copy()
    mov["semana"] = mov["fecha"].dt.to_period("W").dt.start_time
    return mov.groupby(['repuesto','semana'],as_index=False)['cantidad'].sum()


def rellenar_con_ceros(semanal: pd.DataFrame, fin: pd.Timestamp | None = None)-> pd.DataFrame:
    if fin is None:
        fin = semanal['semana'].max()

    inicio = semanal.groupby('repuesto')['semana'].min()

    partes = []

    for repuesto, primera in inicio.items(): 
        semanas = pd.date_range(start = primera, end = fin , freq = "W-MON")
        partes.append(pd.DataFrame({'repuesto':repuesto, 'semana':semanas}))
    esqueleto = pd.concat(partes, ignore_index=True)   
    
    serie = esqueleto.merge(semanal, on=['repuesto','semana'],how='left')
    serie["cantidad"] = serie["cantidad"].fillna(0).astype(int)
    return serie



def rellenar_ceros(semanal: pd.DataFrame, fin: pd.Timestamp | None = None) -> pd.DataFrame:
    """Completa con 0 las semanas sin venta, desde la primera venta de cada repuesto hasta `fin`."""
    if fin is None:
        fin = semanal["semana"].max()

    inicio = semanal.groupby("repuesto")["semana"].min()

    partes = []
    for repuesto, primera in inicio.items():
        semanas = pd.date_range(start=primera, end=fin, freq="W-MON")
        partes.append(pd.DataFrame({"repuesto": repuesto, "semana": semanas}))
    esqueleto = pd.concat(partes, ignore_index=True)

    serie = esqueleto.merge(semanal, on=["repuesto", "semana"], how="left")
    serie["cantidad"] = serie["cantidad"].fillna(0).astype(int)
    return serie



import numpy as np


def etiquetar_repuestos(
    serie: pd.DataFrame,
    min_semanas_con_venta: int = 10,
    umbral_frecuencia: float = 0.25,
) -> pd.DataFrame:
    """Una fila por repuesto: historia, semanas con venta, si entra al modelo y su segmento."""
    serie = serie.copy()
    serie["vendio"] = serie["cantidad"] > 0

    resumen = serie.groupby("repuesto", as_index=False).agg(
        semanas_historia=("semana", "count"),
        semanas_con_venta=("vendio", "sum"),
    )
    resumen["frac_venta"] = resumen["semanas_con_venta"] / resumen["semanas_historia"]
    resumen["entra_modelo"] = resumen["semanas_con_venta"] >= min_semanas_con_venta
    resumen["segmento"] = np.where(
        resumen["frac_venta"] >= umbral_frecuencia, "frecuencia_alta", "intermitente"
    )
    return resumen