from contextlib import nullcontext

import numpy as np
import pandas as pd

from stockifai_forecast.evaluacion import etiquetar_con_train, evaluar_baselines
from stockifai_forecast.modelado import (
    columnas_features, entrenar_modelos, evaluar_modelo, predecir, separar_validacion,
)


def calcular_cortes(
    abt: pd.DataFrame, n_examenes: int = 4, semanas_test: int = 26, horizonte: int = 4
) -> list[pd.Timestamp]:
    """Inicios de ventanas consecutivas, de más antigua a más reciente.

    El último corte coincide con dividir_por_fecha. No descuenta los targets
    faltantes antes de elegir el corte; horizonte solo valida la ventana.
    """
    for nombre, valor in [("n_examenes", n_examenes), ("semanas_test", semanas_test),
                          ("horizonte", horizonte)]:
        if not isinstance(valor, int) or valor < 1:
            raise ValueError(f"{nombre} debe ser un entero positivo")
    if semanas_test <= horizonte:
        raise ValueError("semanas_test debe superar horizonte para tener orígenes evaluables")
    if abt.empty:
        raise ValueError("la ABT no puede estar vacía")
    ultimo = abt["semana"].max() - pd.Timedelta(weeks=semanas_test - 1)
    return [ultimo - pd.Timedelta(weeks=semanas_test * i)
            for i in reversed(range(n_examenes))]


def correr_examen(
    serie: pd.DataFrame,
    abt: pd.DataFrame,
    inicio_test: pd.Timestamp,
    semanas_test: int = 26,
    horizonte: int = 4,
    semanas_val: int = 8,
    semilla: int = 42,
) -> dict:
    """Repite cortes, etiquetas, baselines y L1 para un origen de evaluación.

    Ventana [inicio_test, limite_test), con limite_test exclusivo. Solo evalúa
    orígenes cuyos targets completos terminan dentro de esa misma ventana.
    Si falta prom_8, excluye esa fila para los TRES métodos y lo informa.
    No reajusta con todo train ni guarda modelos. Los índices y etiquetas se
    devuelven para auditar cortes; las entradas nunca se modifican.
    """
    calcular_cortes(abt, n_examenes=1, semanas_test=semanas_test, horizonte=horizonte)
    if not abt.index.is_unique:
        raise ValueError("la ABT debe tener índices únicos para auditar las filas")
    inicio_test = pd.Timestamp(inicio_test)
    if inicio_test.dayofweek != 0:
        raise ValueError("inicio_test debe ser un lunes")
    limite_test = inicio_test + pd.Timedelta(weeks=semanas_test)
    ultima_semana = limite_test - pd.Timedelta(weeks=1)
    if ultima_semana > abt["semana"].max():
        raise ValueError("la ventana de test excede el final de la ABT")

    fin_targets = abt["semana"] + pd.Timedelta(weeks=horizonte)
    train = abt.loc[fin_targets < inicio_test].copy()
    if train.empty:
        raise ValueError("No hay historia suficiente para entrenar este examen")
    ventana = abt.loc[(abt["semana"] >= inicio_test) & (abt["semana"] < limite_test)]
    targets = [f"target_{h}" for h in range(1, horizonte + 1)]
    completas = ventana[targets].notna().all(axis=1)
    dentro = ventana["semana"] + pd.Timedelta(weeks=horizonte) < limite_test
    test = ventana.loc[completas & dentro].copy()
    internas = separar_validacion(train, semanas_val=semanas_val, horizonte=horizonte)
    etiquetas_test = etiquetar_con_train(serie, inicio_test)
    etiquetas_internas = etiquetar_con_train(serie, internas["inicio_val"])

    def unir(tabla, etiquetas):
        return tabla.join(etiquetas.set_index("repuesto"), on="repuesto", validate="many_to_one")

    def universo(tabla):
        return tabla.loc[tabla["entra_modelo"].eq(True).fillna(False)].copy()

    fit = universo(unir(internas["fit"], etiquetas_internas))
    val = universo(unir(internas["val"], etiquetas_internas))
    test = universo(unir(test, etiquetas_test))
    sin_promedio = int(test["prom_8"].isna().sum())
    test = test.loc[test["prom_8"].notna()].copy()
    if fit.empty or val.empty or test.empty:
        raise ValueError("fit, val y test deben tener filas del universo evaluable")

    features = columnas_features(fit)
    baselines = evaluar_baselines(test, horizonte=horizonte)
    modelos = entrenar_modelos(fit, val, features, "l1", horizonte=horizonte, semilla=semilla)
    predicciones = predecir(modelos, test, features)
    modelo = evaluar_modelo(test, predicciones, "lightgbm_l1")
    metricas = pd.concat([baselines, modelo], ignore_index=True)
    # La máscara común garantiza las mismas claves, no solo igual cantidad.
    referencias = baselines.loc[baselines["baseline"] == "cero"].set_index(
        ["horizonte", "agrupacion", "grupo"]
    )["n_filas"].sort_index()
    for nombre in ("prom_8", "lightgbm_l1"):
        usadas = metricas.loc[metricas["baseline"] == nombre].set_index(
            ["horizonte", "agrupacion", "grupo"]
        )["n_filas"].sort_index()
        pd.testing.assert_series_equal(referencias, usadas)

    elegibles = etiquetas_test.loc[etiquetas_test["entra_modelo"]]
    conteos = {
        "total": len(elegibles),
        "segmento": elegibles.groupby("segmento", observed=True).size().to_dict(),
        "grupo_historia": elegibles.groupby("grupo_historia", observed=True).size().to_dict(),
    }
    fechas = {
        nombre: {"inicio": tabla["semana"].min(), "fin": tabla["semana"].max()}
        for nombre, tabla in [("train", train), ("fit", fit), ("val", val), ("test", test)]
    }
    return {
        "inicio_test": inicio_test, "fin_test": ultima_semana, "limite_test": limite_test,
        "inicio_val": internas["inicio_val"], "fechas": fechas,
        "filas": {"train": len(train), "fit": len(fit), "val": len(val), "test": len(test),
                  "test_ventana_sin_universo": len(ventana), "test_excluidas_sin_prom_8": sin_promedio},
        "repuestos": conteos,
        "indices": {"train": train.index.copy(), "fit": fit.index.copy(),
                    "val": val.index.copy(), "test": test.index.copy()},
        "etiquetas_test": etiquetas_test, "etiquetas_fit_val": etiquetas_internas,
        "mejores_iteraciones": {h: int(m.best_iteration_) for h, m in modelos.items()},
        "mae_validacion": {h: float(m.best_score_["valid_0"]["l1"]) for h, m in modelos.items()},
        "metricas": metricas,
    }


def _registrar_examen(mlflow, resultado):
    parametros = {
        "inicio_test": str(resultado["inicio_test"].date()),
        "fin_test": str(resultado["fin_test"].date()),
        "inicio_val": str(resultado["inicio_val"].date()),
        "repuestos_universo": resultado["repuestos"]["total"],
        **{f"filas_{k}": v for k, v in resultado["filas"].items()},
    }
    for parte, limites in resultado["fechas"].items():
        for extremo, fecha in limites.items():
            parametros[f"{parte}_{extremo}"] = str(fecha.date())
    for dimension in ("segmento", "grupo_historia"):
        for grupo, n in resultado["repuestos"][dimension].items():
            parametros[f"repuestos_{dimension}_{str(grupo).replace('+', '_mas')}"] = n
    mlflow.log_params(parametros)
    valores = {}
    for fila in resultado["metricas"].itertuples(index=False):
        grupo = str(fila.grupo).replace("+", "_mas")
        base = f"test/{fila.baseline}/{fila.agrupacion}/{grupo}/h{fila.horizonte}"
        for campo in ("mae", "n_candidatas", "n_filas", "n_excluidas", "n_sin_target"):
            valor = getattr(fila, campo)
            if pd.notna(valor):
                valores[f"{base}/{campo}"] = float(valor)
    valores.update({f"mejor_iteracion/h{h}": v for h, v in resultado["mejores_iteraciones"].items()})
    valores.update({f"validacion/mae_h{h}": v for h, v in resultado["mae_validacion"].items()})
    mlflow.log_metrics(valores)
    mlflow.log_text(resultado["metricas"].to_csv(index=False), "metricas.csv")


def correr_backtesting(
    serie: pd.DataFrame,
    abt: pd.DataFrame,
    n_examenes: int = 4,
    semanas_test: int = 26,
    horizonte: int = 4,
    semanas_val: int = 8,
    semilla: int = 42,
    registrar_mlflow: bool = False,
) -> pd.DataFrame:
    """Métricas largas con examen 1=el más antiguo.

    attrs['examenes'] contiene fechas, conteos e iteraciones sin retener modelos
    ni las tablas grandes. MLflow es opcional y usa el tracking configurado por
    el llamador. Si se habilita, crea un padre y un hijo por examen.
    """
    cortes = calcular_cortes(abt, n_examenes, semanas_test, horizonte)
    if registrar_mlflow:
        import mlflow
    padre = mlflow.start_run(run_name="backtesting") if registrar_mlflow else nullcontext()
    tablas, detalles, runs = [], [], {}
    with padre as run_padre:
        if registrar_mlflow:
            runs["padre"] = run_padre.info.run_id
            mlflow.log_params({"n_examenes": n_examenes, "semanas_test": semanas_test,
                               "horizonte": horizonte, "semanas_val": semanas_val,
                               "objective": "l1", "learning_rate": 0.05, "n_estimators": 2000,
                               "early_stopping_rounds": 100, "semilla": semilla})
        for examen, inicio in enumerate(cortes, start=1):
            hijo = mlflow.start_run(run_name=f"examen_{examen}", nested=True) if registrar_mlflow else nullcontext()
            with hijo as run_hijo:
                resultado = correr_examen(serie, abt, inicio, semanas_test, horizonte, semanas_val, semilla)
                if registrar_mlflow:
                    runs[f"examen_{examen}"] = run_hijo.info.run_id
                    _registrar_examen(mlflow, resultado)
                tabla = resultado["metricas"].copy()
                tabla.insert(0, "examen", examen)
                tabla.insert(1, "inicio_test", inicio)
                tablas.append(tabla)
                detalles.append({k: v for k, v in resultado.items()
                                 if k not in ("indices", "etiquetas_test", "etiquetas_fit_val", "metricas")})
        metricas = pd.concat(tablas, ignore_index=True)
        if registrar_mlflow:
            mlflow.log_text(resumir(metricas).to_csv(index=False), "resumen.csv")
    metricas.attrs["examenes"] = detalles
    metricas.attrs["run_ids"] = runs
    return metricas


def resumir(metricas: pd.DataFrame) -> pd.DataFrame:
    """Una fila por grupo y método; promedia horizontes y luego exámenes.

    mae_min/max son extremos de los MAE medios POR EXAMEN, no de horizontes.
    Cada examen pesa igual, independientemente de su número de filas/SKUs.
    Un empate numérico (atol=1e-10, rtol=0) cuenta como ganado por ambos.
    Las categorías ausentes no se inventan con MAE cero.
    """
    columnas = ["agrupacion", "grupo", "baseline", "mae_promedio", "mae_min",
                "mae_max", "examenes_ganados", "n_examenes"]
    if metricas.empty:
        return pd.DataFrame(columns=columnas)
    por_examen = (metricas.groupby(
        ["examen", "agrupacion", "grupo", "baseline"], dropna=False, observed=True
    )["mae"].mean().reset_index())
    minimo = por_examen.groupby(["examen", "agrupacion", "grupo"], dropna=False)["mae"].transform("min")
    por_examen["gano"] = np.isclose(por_examen["mae"], minimo, rtol=0, atol=1e-10)
    resumen = por_examen.groupby(["agrupacion", "grupo", "baseline"], dropna=False, observed=True).agg(
        mae_promedio=("mae", "mean"), mae_min=("mae", "min"), mae_max=("mae", "max"),
        examenes_ganados=("gano", "sum"), n_examenes=("mae", "count"),
    ).reset_index()
    return resumen[columnas]
