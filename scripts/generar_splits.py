# -*- coding: utf-8 -*-
"""
Genera train/validation/test a partir de gold_set_final.csv.

Misma metodología que el notebook 01 (celda 11):
  - Split POR review_uid (evita fuga: una reseña nunca queda en dos splits).
  - Estratificado por la etiqueta dominante de cada reseña (si hay soporte).
  - Proporción 70 / 15 / 15, semilla fija.
  - Escribe en data/ y en outputs/ (el notebook 02 lee desde data/).
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST = 0.70, 0.15, 0.15

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORTS_DIR = OUTPUT_DIR / "reports"
for d in (DATA_DIR, OUTPUT_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

VALID_LABELS = ["negativo", "neutro", "positivo"]

gold = pd.read_csv(DATA_DIR / "gold_set_final.csv", encoding="utf-8-sig")

# Solo filas con etiqueta válida: las no etiquetadas no sirven para entrenar/evaluar
# y el notebook 02 las rechaza.
n_total = len(gold)
gold = gold[gold["label"].isin(VALID_LABELS)].copy()
print(f"Filas con label válido: {len(gold)} de {n_total} ({n_total - len(gold)} sin etiqueta, descartadas)")

# Una etiqueta dominante por review_uid para estratificar a nivel de reseña.
uid_labels = (
    gold.groupby("review_uid")["label"]
    .agg(lambda x: x.value_counts().index[0])
    .reset_index()
)
unique_uids = uid_labels["review_uid"].to_numpy(dtype=object)
uid_y = uid_labels["label"].to_numpy(dtype=object)

can_stratify = pd.Series(uid_y).value_counts().min() >= 2
stratify_y = uid_y if can_stratify else None

train_uids, temp_uids, _, y_temp = train_test_split(
    unique_uids, uid_y, test_size=(1 - SPLIT_TRAIN), random_state=SEED, stratify=stratify_y,
)
temp_stratify = y_temp if pd.Series(y_temp).value_counts().min() >= 2 else None
validation_ratio_inside_temp = SPLIT_VALIDATION / (SPLIT_VALIDATION + SPLIT_TEST)
validation_uids, test_uids = train_test_split(
    temp_uids, test_size=(1 - validation_ratio_inside_temp), random_state=SEED, stratify=temp_stratify,
)

train_df = gold[gold["review_uid"].isin(train_uids)].copy()
validation_df = gold[gold["review_uid"].isin(validation_uids)].copy()
test_df = gold[gold["review_uid"].isin(test_uids)].copy()

# Verificación de fuga
assert set(train_df["review_uid"]).isdisjoint(validation_df["review_uid"]), "Leakage train-validation."
assert set(train_df["review_uid"]).isdisjoint(test_df["review_uid"]), "Leakage train-test."
assert set(validation_df["review_uid"]).isdisjoint(test_df["review_uid"]), "Leakage validation-test."
# Cobertura total
assert len(train_df) + len(validation_df) + len(test_df) == len(gold), "Las filas no suman el gold set."

for d in (DATA_DIR, OUTPUT_DIR):
    train_df.to_csv(d / "train.csv", index=False, encoding="utf-8-sig")
    validation_df.to_csv(d / "validation.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(d / "test.csv", index=False, encoding="utf-8-sig")

print(f"Gold set (etiquetado): {len(gold)} filas | {gold['review_uid'].nunique()} reseñas")
print("-" * 60)
for name, df in [("train", train_df), ("validation", validation_df), ("test", test_df)]:
    dist = df["label"].value_counts().reindex(VALID_LABELS, fill_value=0)
    pct = len(df) / len(gold) * 100
    print(f"{name:11s} {len(df):4d} filas ({pct:4.1f}%) | {df['review_uid'].nunique():4d} reseñas | "
          + "  ".join(f"{l}={int(dist[l])}" for l in VALID_LABELS))
print("-" * 60)
print("OK: sin fuga por review_uid, cobertura total del gold set.")
print("Escrito en: data/ y outputs/")

summary = pd.DataFrame([
    {"split": n, "filas": len(d), "review_uid_unicos": d["review_uid"].nunique()}
    for n, d in [("train", train_df), ("validation", validation_df), ("test", test_df)]
])
summary.to_csv(REPORTS_DIR / "resumen_splits_gold_set.csv", index=False, encoding="utf-8-sig")
