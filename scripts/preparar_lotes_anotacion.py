# -*- coding: utf-8 -*-
# ============================================================
# Prepara lotes de anotacion para 3 anotadores con SOLAPAMIENTO para kappa.
# ------------------------------------------------------------
# Toma la cola de candidatos negativos y la reparte en 3 muestras (una por
# anotador) replicando el flujo existente (muestra_anotador_1/2/3). Un subconjunto
# SOLAPADO es anotado por los 3 para recalcular Fleiss/Cohen kappa.
#
# Importante (anti-anclaje): las muestras que ven los anotadores NO incluyen la
# sugerencia heuristica, ni estrellas, ni pistas lexicas, ni la calidad de senal.
# Solo: annotation_id, review_uid, destination, language_review, aspecto,
# text_clean, input_modelo y una columna 'label' VACIA que el anotador llena con
# positivo / neutro / negativo.
#
# Toda la metadata heuristica se conserva aparte en _clave_maestra.csv para
# trazabilidad y analisis posterior (NO se entrega a los anotadores).
#
# Salidas (en data/anotacion_v2/):
#   muestra_anotador_1.csv, muestra_anotador_2.csv, muestra_anotador_3.csv
#   _clave_maestra.csv         (mapa id -> overlap, anotadores, metadata)
#   _resumen_lotes.csv         (conteos por anotador y aspecto)
# ============================================================
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT = DATA / "anotacion_v2"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_OVERLAP = 120   # items anotados por los 3 (para kappa)

cand = pd.read_csv(DATA / "candidatos_negativos_para_anotacion.csv", encoding="utf-8-sig")
cand["annotation_id"] = cand["review_uid"].astype(str) + "__" + cand["aspecto"].astype(str)
cand = cand.drop_duplicates(subset=["annotation_id"]).reset_index(drop=True)

# Mezcla reproducible
cand = cand.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

# --- Solapamiento estratificado por aspecto ---
frac_overlap = min(0.95, N_OVERLAP / len(cand))
overlap = (cand.groupby("aspecto", group_keys=False)
                .apply(lambda g: g.sample(frac=frac_overlap, random_state=SEED)))
overlap_ids = set(overlap["annotation_id"])
resto = cand[~cand["annotation_id"].isin(overlap_ids)].copy()

# --- Reparto del resto en 3 lotes disjuntos, balanceado por aspecto (round-robin) ---
asignacion = {1: [], 2: [], 3: []}
resto = resto.sort_values(["aspecto", "annotation_id"]).reset_index(drop=True)
for i, row in enumerate(resto.itertuples(index=False)):
    asignacion[(i % 3) + 1].append(row.annotation_id)

COLS_ANOTADOR = ["annotation_id", "review_uid", "destination", "language_review",
                 "aspecto", "text_clean", "input_modelo", "label"]


def construir_muestra(ids_unicos, anotador):
    ids = list(overlap_ids) + list(ids_unicos)           # solapados + unicos
    m = cand[cand["annotation_id"].isin(ids)].copy()
    m["label"] = ""                                       # el anotador la llena
    m = m.sample(frac=1.0, random_state=SEED + anotador)  # orden mezclado
    return m[COLS_ANOTADOR]


resumen = []
for a in (1, 2, 3):
    m = construir_muestra(asignacion[a], a)
    m.to_csv(OUT / f"muestra_anotador_{a}.csv", index=False, encoding="utf-8-sig")
    resumen.append({"anotador": a, "total": len(m), "solapados": len(overlap_ids),
                    "unicos": len(asignacion[a])})
    for asp, n in m["aspecto"].value_counts().items():
        resumen.append({"anotador": a, "aspecto": asp, "n": int(n)})

# --- Clave maestra (trazabilidad, NO para anotadores) ---
clave = cand[["annotation_id", "review_uid", "destination", "aspecto", "language_review",
              "stars", "sentiment_by_stars", "n_cues_neg", "cues_neg", "calidad_senal"]].copy()
clave["en_overlap"] = clave["annotation_id"].isin(overlap_ids)
uid2anot = {}
for a in (1, 2, 3):
    for i in asignacion[a]:
        uid2anot[i] = str(a)
clave["anotador_asignado"] = clave["annotation_id"].map(
    lambda x: "1;2;3 (overlap)" if x in overlap_ids else uid2anot.get(x, ""))
clave.to_csv(OUT / "_clave_maestra.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(resumen).to_csv(OUT / "_resumen_lotes.csv", index=False, encoding="utf-8-sig")

print(f"Candidatos totales: {len(cand)}")
print(f"Overlap (3 anotadores, para kappa): {len(overlap_ids)}")
print(f"Unicos por anotador: " + ", ".join(str(len(asignacion[a])) for a in (1, 2, 3)))
for a in (1, 2, 3):
    n = len(overlap_ids) + len(asignacion[a])
    print(f"  Anotador {a}: {n} items a etiquetar")
print("Total etiquetas a producir:", len(overlap_ids) * 3 + sum(len(asignacion[a]) for a in (1, 2, 3)))
print("Salidas en:", OUT)
