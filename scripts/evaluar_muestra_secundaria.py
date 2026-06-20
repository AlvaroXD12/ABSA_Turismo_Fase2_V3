# -*- coding: utf-8 -*-
# R18 (versión CON etiquetas): cuando data/muestra_evaluacion_secundaria.csv esté
# anotada (columna 'label' llena), evalúa el modelo XLM-R en esa muestra de
# distribución REAL y la compara contra el test estratificado.
from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

BASE = Path(__file__).resolve().parent.parent
DATA, REP = BASE / "data", BASE / "outputs" / "reports"
L = ["negativo", "neutro", "positivo"]

mm = pd.read_csv(DATA / "muestra_evaluacion_secundaria.csv", encoding="utf-8-sig")
mm["label"] = mm["label"].astype(str).str.lower().str.strip()
anot = mm[mm["label"].isin(L)].copy()
if len(anot) < 50:
    print(f"La muestra aún no está anotada ({len(anot)} etiquetas válidas). Llena la columna 'label' primero.")
    raise SystemExit

pred = pd.read_csv(BASE / "outputs/predictions/predicciones_corpus_v4.csv", encoding="utf-8-sig")
ev = anot.merge(pred.rename(columns={"label_pred": "pred"})[["review_uid", "aspecto", "pred"]],
                on=["review_uid", "aspecto"], how="left").dropna(subset=["pred"])
pr, rc, f1, _ = precision_recall_fscore_support(ev["label"], ev["pred"], labels=L, average=None, zero_division=0)
_, _, mf1, _ = precision_recall_fscore_support(ev["label"], ev["pred"], labels=L, average="macro", zero_division=0)
res = {"n": len(ev), "f1_macro": round(mf1, 4), "accuracy": round(accuracy_score(ev["label"], ev["pred"]), 4)}
for i, l in enumerate(L): res[f"f1_{l}"] = round(f1[i], 4); res[f"recall_{l}"] = round(rc[i], 4)
print("=== Evaluación secundaria CON etiquetas (distribución real, NO balanceada) ===")
print("Distribución real de la muestra:", ev["label"].value_counts().to_dict())
for k, v in res.items(): print(f"  {k}: {v}")
pd.DataFrame([res]).to_csv(REP / "eval_secundaria_con_etiquetas.csv", index=False, encoding="utf-8-sig")
print("\nNota: NO reemplaza el F1-macro del test estratificado (0.709); verifica coherencia en producción.")
