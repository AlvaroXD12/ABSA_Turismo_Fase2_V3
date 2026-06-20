# -*- coding: utf-8 -*-
# ============================================================
# R10-R11: Validación del diccionario de aspectos (etapa de reglas del flujo híbrido).
#  - Cobertura: % de reseñas con >=1 aspecto detectado (computable ya).
#  - FP/FN por aspecto: requiere una muestra anotada de PRESENCIA de aspectos
#    (multi-etiqueta). Genera la plantilla; al anotarse, calcula precisión/recall/F1
#    de la detección por reglas, global y por aspecto.
# La clasificación de POLARIDAD la hace XLM-R+TextCNN (no el diccionario).
#
# Salidas:
#   outputs/reports/diccionario_cobertura.csv
#   data/muestra_validacion_diccionario.csv   (plantilla multi-etiqueta, vacía)
#   outputs/reports/diccionario_fp_fn.csv      (cuando la muestra esté anotada)
# ============================================================
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA, REP = BASE / "data", BASE / "outputs" / "reports"
ASPECTOS = ["atractivos","costos","seguridad","accesibilidad","limpieza",
            "atencion_servicio","gastronomia","alojamiento","clima","aforo_multitudes"]
N_MUESTRA = 150

corpus = pd.read_csv(DATA / "tourism_reviews_clean.csv", encoding="utf-8-sig")
absa = pd.read_csv(BASE / "outputs/predictions/tourism_reviews_clean_absa_ready.csv", encoding="utf-8-sig")
n_total = corpus["review_uid"].nunique()

# ---------- Cobertura (ya computable) ----------
cov_rows = [{"aspecto": "TODOS (>=1 aspecto)", "reseñas_detectadas": absa["review_uid"].nunique(),
             "%_corpus": round(absa["review_uid"].nunique()/n_total*100, 1)}]
for a in ASPECTOS:
    n = absa[absa.aspecto == a]["review_uid"].nunique()
    cov_rows.append({"aspecto": a, "reseñas_detectadas": n, "%_corpus": round(n/n_total*100, 1)})
pd.DataFrame(cov_rows).to_csv(REP / "diccionario_cobertura.csv", index=False, encoding="utf-8-sig")
print(f"Cobertura del léxico (>=1 aspecto): {absa['review_uid'].nunique()/n_total*100:.1f}% de {n_total} reseñas")

# ---------- FP/FN: plantilla o evaluación ----------
muestra_path = DATA / "muestra_validacion_diccionario.csv"
asp_cols = [f"asp_{a}" for a in ASPECTOS]

def ya_anotada(df):
    if not all(c in df.columns for c in asp_cols): return False
    return df[asp_cols].notna().any(axis=1).sum() >= 30

if muestra_path.exists() and ya_anotada(pd.read_csv(muestra_path, encoding="utf-8-sig")):
    mm = pd.read_csv(muestra_path, encoding="utf-8-sig")
    mm = mm[mm[asp_cols].notna().any(axis=1)].copy()
    for c in asp_cols: mm[c] = pd.to_numeric(mm[c], errors="coerce").fillna(0).astype(int)
    det = absa.groupby("review_uid")["aspecto"].apply(set).to_dict()   # lo que el léxico detectó
    filas = []
    for a in ASPECTOS:
        tp = fp = fn = 0
        for _, r in mm.iterrows():
            humano = r[f"asp_{a}"] == 1
            lexico = a in det.get(r["review_uid"], set())
            tp += int(humano and lexico); fp += int(lexico and not humano); fn += int(humano and not lexico)
        prec = tp/(tp+fp) if tp+fp else 0.0; rec = tp/(tp+fn) if tp+fn else 0.0
        f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0
        filas.append({"aspecto": a, "TP": tp, "FP": fp, "FN": fn,
                      "precision": round(prec,3), "recall": round(rec,3), "f1": round(f1,3)})
    res = pd.DataFrame(filas); res.to_csv(REP / "diccionario_fp_fn.csv", index=False, encoding="utf-8-sig")
    print("=== FP/FN de la detección por reglas (sobre muestra anotada) ==="); print(res.to_string(index=False))
    print("Macro precisión/recall/F1:", res[["precision","recall","f1"]].mean().round(3).to_dict())
else:
    sample = corpus.drop_duplicates("review_uid").sample(min(N_MUESTRA, n_total), random_state=42)
    t = sample[["review_uid", "destination", "language_review", "text_clean"]].copy()
    for c in asp_cols: t[c] = ""   # el anotador marca 1 si el aspecto SÍ se discute (sin anclaje)
    t.to_csv(muestra_path, index=False, encoding="utf-8-sig")
    print(f"Plantilla de validación generada: {len(t)} reseñas -> {muestra_path}")
    print("El anotador marca 1/0 en las 10 columnas asp_* según qué aspectos se discuten realmente.")
