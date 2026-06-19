# -*- coding: utf-8 -*-
# ============================================================
# Generador de la matriz destino-aspecto-sentimiento (spec R12-R16).
# ------------------------------------------------------------
# Construye la matriz analitica que consumira la Fase 3. Cada celda (destino x
# aspecto) trae: conteos y proporciones por polaridad, score [-1,1], etiqueta
# dominante, nivel de evidencia, conflict_flag y confianza, segun las reglas de
# la especificacion.
#
# ENTRADA: un dataframe de predicciones con columnas
#   review_uid, destination, aspecto, label_pred  (label_pred in pos/neu/neg)
# Por defecto usa un STAND-IN (polaridad por estrellas) para VALIDAR la logica
# sin modelo. Cuando exista el modelo entrenado, pasar --pred <csv> con la columna
# label_pred = prediccion del modelo sobre el corpus (absa_ready).
#
# SALIDA: outputs/matrices/matriz_destino_aspecto_sentimiento.csv (+ _largo.csv)
# ============================================================
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUTM = BASE / "outputs" / "matrices"
OUTM.mkdir(parents=True, exist_ok=True)
L = ["negativo", "neutro", "positivo"]

# Taxonomia fija de aspectos (spec). Se generan TODAS las celdas destino x aspecto,
# incluso sin menciones (-> "sin datos"), para no ocultar vacios a la Fase 3.
ASPECTOS = ["atractivos", "costos", "seguridad", "accesibilidad", "limpieza",
            "atencion_servicio", "gastronomia", "alojamiento", "clima", "aforo_multitudes"]


def nivel_evidencia(n):
    if n == 0:  return "sin datos"
    if n <= 4:  return "evidencia insuficiente"
    if n <= 9:  return "baja evidencia"
    return "evidencia suficiente"


def construir_celda(g):
    n = len(g)
    npos = int((g["label_pred"] == "positivo").sum())
    nneu = int((g["label_pred"] == "neutro").sum())
    nneg = int((g["label_pred"] == "negativo").sum())
    ntot = npos + nneu + nneg
    ppos = npos / ntot if ntot else 0.0
    pneu = nneu / ntot if ntot else 0.0
    pneg = nneg / ntot if ntot else 0.0
    score = (npos - nneg) / ntot if ntot else 0.0
    # Conflicto (spec R15b): n>=5, ppos>=.25, pneg>=.25, |ppos-pneg|<.15
    conflict = int(n >= 5 and ppos >= 0.25 and pneg >= 0.25 and abs(ppos - pneg) < 0.15)
    # Etiqueta dominante (interpretativa); si conflicto -> mixta/conflictiva
    if ntot == 0:
        etiqueta = "sin datos"
    elif conflict:
        etiqueta = "mixta/conflictiva"
    else:
        etiqueta = L[int(np.argmax([nneg, nneu, npos]))]
    # Confianza (spec R15)
    conf_ev = min(1.0, n / 10.0)
    confianza = conf_ev * 0.65 if conflict else conf_ev
    return pd.Series({
        "n_menciones": n,
        "n_resenas_unicas": int(g["review_uid"].nunique()),
        "n_positivo": npos, "n_neutro": nneu, "n_negativo": nneg,
        "prop_positivo": round(ppos, 4), "prop_neutro": round(pneu, 4), "prop_negativo": round(pneg, 4),
        "score_sentimiento": round(score, 4),
        "etiqueta_dominante": etiqueta,
        "nivel_evidencia": nivel_evidencia(n),
        "conflict_flag": conflict,
        "confianza": round(confianza, 4),
    })


def build_matrix(pred):
    pred = pred[pred["label_pred"].isin(L)].copy()
    destinos = sorted(pred["destination"].dropna().unique())
    # Grid completo destino x aspecto
    grid = pd.MultiIndex.from_product([destinos, ASPECTOS], names=["destination", "aspecto"]).to_frame(index=False)
    agg = (pred.groupby(["destination", "aspecto"]).apply(construir_celda).reset_index())
    matriz = grid.merge(agg, on=["destination", "aspecto"], how="left")
    # Celdas sin menciones -> "sin datos"
    cnt = ["n_menciones", "n_resenas_unicas", "n_positivo", "n_neutro", "n_negativo"]
    matriz[cnt] = matriz[cnt].fillna(0).astype(int)
    prop = ["prop_positivo", "prop_neutro", "prop_negativo", "score_sentimiento", "confianza"]
    matriz[prop] = matriz[prop].fillna(0.0)
    matriz["conflict_flag"] = matriz["conflict_flag"].fillna(0).astype(int)
    matriz["etiqueta_dominante"] = matriz["etiqueta_dominante"].fillna("sin datos")
    matriz["nivel_evidencia"] = matriz["nivel_evidencia"].fillna("sin datos")
    return matriz.sort_values(["destination", "aspecto"]).reset_index(drop=True)


def cargar_standin():
    """Predicciones STAND-IN: polaridad por estrellas (para validar la logica sin modelo)."""
    absa = pd.read_csv(BASE / "outputs/predictions/tourism_reviews_clean_absa_ready.csv", encoding="utf-8-sig")
    clean = pd.read_csv(DATA / "tourism_reviews_clean.csv", encoding="utf-8-sig",
                        usecols=["review_uid", "sentiment_by_stars"])
    df = absa.merge(clean, on="review_uid", how="left")
    df["label_pred"] = df["sentiment_by_stars"].astype(str).str.lower().str.strip()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", help="CSV con review_uid,destination,aspecto,label_pred (predicciones del modelo)")
    args = ap.parse_args()
    if args.pred:
        pred = pd.read_csv(args.pred, encoding="utf-8-sig")
        fuente = f"predicciones del modelo ({Path(args.pred).name})"
        sufijo = ""
    else:
        pred = cargar_standin()
        fuente = "STAND-IN (polaridad por estrellas) — solo valida la logica, NO es la matriz final"
        sufijo = "_standin"

    print("Fuente de polaridad:", fuente)
    print("Pares (reseña x aspecto):", len(pred))
    matriz = build_matrix(pred)
    out = OUTM / f"matriz_destino_aspecto_sentimiento{sufijo}.csv"
    matriz.to_csv(out, index=False, encoding="utf-8-sig")
    print("Matriz ->", out, "| celdas:", len(matriz))
    print("\nNiveles de evidencia:", matriz["nivel_evidencia"].value_counts().to_dict())
    print("Celdas con conflicto:", int(matriz["conflict_flag"].sum()))
    print("\nEjemplo (Machu Picchu):")
    cols = ["aspecto", "n_menciones", "n_negativo", "n_neutro", "n_positivo",
            "score_sentimiento", "etiqueta_dominante", "nivel_evidencia", "conflict_flag", "confianza"]
    mp = matriz[matriz["destination"].str.contains("Machu", na=False)]
    if len(mp):
        print(mp[cols].to_string(index=False))


if __name__ == "__main__":
    main()
