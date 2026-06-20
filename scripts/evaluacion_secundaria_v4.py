# -*- coding: utf-8 -*-
# ============================================================
# R17-R18: Evaluación SECUNDARIA del modelo final (XLM-R) en DISTRIBUCIÓN REAL.
# No reemplaza el F1-macro del test estratificado; verifica COHERENCIA cuando
# predominan reseñas positivas (condición de producción).
#
# Dos entregables:
#  A) Análisis de coherencia AHORA (sin anotar): usa las predicciones del modelo
#     sobre el corpus + la polaridad por estrellas (referencia DÉBIL, a nivel de
#     reseña). Mide distribución real predicha, acuerdo vs estrellas y por aspecto.
#  B) Muestra ANOTABLE (>=300, ideal 500) aleatoria, NO balanceada, proporcional
#     por destino e idioma, para la versión con etiquetas (la llenan los anotadores).
#
# Salidas:
#   outputs/reports/eval_secundaria_distribucion.csv
#   outputs/reports/eval_secundaria_acuerdo_estrellas.csv
#   outputs/reports/eval_secundaria_por_aspecto.csv
#   data/muestra_evaluacion_secundaria.csv   (anotable, label vacía)
# ============================================================
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score

BASE = Path(__file__).resolve().parent.parent
DATA, REP = BASE / "data", BASE / "outputs" / "reports"
L = ["negativo", "neutro", "positivo"]
N_MUESTRA = 500   # objetivo de la muestra anotable (>=300)

pred = pd.read_csv(BASE / "outputs/predictions/predicciones_corpus_v4.csv", encoding="utf-8-sig")
clean = pd.read_csv(DATA / "tourism_reviews_clean.csv", encoding="utf-8-sig",
                    usecols=["review_uid", "stars", "sentiment_by_stars", "language_review", "source"])
absa = pd.read_csv(BASE / "outputs/predictions/tourism_reviews_clean_absa_ready.csv", encoding="utf-8-sig")

df = pred.merge(clean, on="review_uid", how="left")
df["label_pred"] = df["label_pred"].astype(str).str.lower().str.strip()
df["sentiment_by_stars"] = df["sentiment_by_stars"].astype(str).str.lower().str.strip()
print(f"Pares reseña×aspecto en el corpus (distribución REAL): {len(df)}")

# ---------- A) Coherencia (sin anotación) ----------
dist = pd.DataFrame({
    "predicho_XLMR_n": df["label_pred"].value_counts().reindex(L, fill_value=0),
    "predicho_XLMR_%": (df["label_pred"].value_counts(normalize=True).reindex(L, fill_value=0) * 100).round(1),
    "por_estrellas_n": df["sentiment_by_stars"].value_counts().reindex(L, fill_value=0),
    "por_estrellas_%": (df["sentiment_by_stars"].value_counts(normalize=True).reindex(L, fill_value=0) * 100).round(1),
})
dist.to_csv(REP / "eval_secundaria_distribucion.csv", encoding="utf-8-sig")
print("\n=== Distribución de polaridad en distribución REAL ==="); print(dist.to_string())

# Acuerdo XLM-R vs estrellas (referencia DÉBIL: estrellas = reseña completa, no aspecto)
m = df.dropna(subset=["sentiment_by_stars"])
m = m[m["sentiment_by_stars"].isin(L)]
acc = accuracy_score(m["sentiment_by_stars"], m["label_pred"])
kap = cohen_kappa_score(m["sentiment_by_stars"], m["label_pred"], labels=L)
cm = confusion_matrix(m["sentiment_by_stars"], m["label_pred"], labels=L)
cmdf = pd.DataFrame(cm, index=[f"estrellas_{l}" for l in L], columns=[f"pred_{l}" for l in L])
cmdf.to_csv(REP / "eval_secundaria_acuerdo_estrellas.csv", encoding="utf-8-sig")
print(f"\nAcuerdo XLM-R vs estrellas (proxy débil): accuracy={acc:.3f} | Cohen kappa={kap:.3f}")
print("Matriz (filas=estrellas, cols=predicho XLM-R):"); print(cmdf.to_string())

# Por aspecto: distribución predicha
asp = df.groupby("aspecto")["label_pred"].value_counts(normalize=True).unstack().reindex(columns=L).fillna(0).round(3)
asp["n"] = df.groupby("aspecto").size()
asp.to_csv(REP / "eval_secundaria_por_aspecto.csv", encoding="utf-8-sig")
print("\n=== Distribución predicha por aspecto (proporción) ==="); print(asp.to_string())

# Por idioma
print("\n=== Predicho por idioma (%) ===")
print((df.groupby("language_review")["label_pred"].value_counts(normalize=True).unstack().reindex(columns=L).fillna(0)*100).round(1).to_string())

# ---------- B) Muestra ANOTABLE (proporcional destino×idioma, NO balanceada) ----------
df["annotation_id"] = df["review_uid"].astype(str) + "__" + df["aspecto"].astype(str)
base = df.merge(absa[["review_uid", "aspecto", "text_clean", "input_modelo"]], on=["review_uid", "aspecto"], how="left")
frac = min(0.95, N_MUESTRA / len(base))
muestra = base.groupby(["destination", "language_review"], group_keys=False).sample(frac=frac, random_state=42)
cols = ["annotation_id", "review_uid", "destination", "language_review", "aspecto", "text_clean", "input_modelo"]
muestra = muestra[cols].copy(); muestra["label"] = ""   # la llenan los anotadores (sin anclaje)
muestra.to_csv(DATA / "muestra_evaluacion_secundaria.csv", index=False, encoding="utf-8-sig")
print(f"\nMuestra anotable (no balanceada): {len(muestra)} instancias -> data/muestra_evaluacion_secundaria.csv")
print("Distribución de la muestra por idioma:", muestra["language_review"].value_counts().to_dict())
