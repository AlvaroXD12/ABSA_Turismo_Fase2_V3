# -*- coding: utf-8 -*-
# ============================================================
# (RE)Genera los lotes de anotacion de los 707 CANDIDATOS NEGATIVOS.
# ------------------------------------------------------------
# Esta es la ampliacion de negativos que quedo PENDIENTE (la entrega anterior
# re-anoto el gold existente, no estos candidatos). Para evitar otra confusion,
# las salidas van a una carpeta y nombres distintos y EXPLICITOS:
#   data/anotacion_negativos_v2/lote_negativos_anotador_{1,2,3}.csv
#
# Excluye candidatos cuyo (review_uid, aspecto) ya fue anotado en la re-anotacion
# triple (data/anotacion_v2) o ya esta en el gold de consenso.
# Mantiene el diseno anti-anclaje (label vacia, sin estrellas/pistas/sugerencia)
# y un subconjunto solapado para kappa.
# ============================================================
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT = DATA / "anotacion_negativos_v2"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
N_OVERLAP = 100

cand = pd.read_csv(DATA / "candidatos_negativos_para_anotacion.csv", encoding="utf-8-sig")
cand["annotation_id"] = cand["review_uid"].astype(str) + "__" + cand["aspecto"].astype(str)
cand = cand.drop_duplicates("annotation_id")

# Excluir lo ya anotado (re-anotacion triple + gold consenso)
ya = set()
for p in [DATA / "anotacion_v2" / "muestra_anotador_1.csv", DATA / "gold_set_consenso_v2.csv"]:
    if p.exists():
        d = pd.read_csv(p, encoding="utf-8-sig")
        if "annotation_id" not in d.columns:
            d["annotation_id"] = d["review_uid"].astype(str) + "__" + d["aspecto"].astype(str)
        ya |= set(d["annotation_id"])
antes = len(cand)
cand = cand[~cand["annotation_id"].isin(ya)].reset_index(drop=True)
print(f"Candidatos: {antes} -> {len(cand)} (excluidos {antes-len(cand)} ya anotados)")

cand = cand.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
frac_ov = min(0.95, N_OVERLAP / len(cand))
overlap = cand.groupby("aspecto", group_keys=False).apply(lambda g: g.sample(frac=frac_ov, random_state=SEED))
overlap_ids = set(overlap["annotation_id"])
resto = cand[~cand["annotation_id"].isin(overlap_ids)].sort_values(["aspecto", "annotation_id"]).reset_index(drop=True)

asign = {1: [], 2: [], 3: []}
for i, row in enumerate(resto.itertuples(index=False)):
    asign[(i % 3) + 1].append(row.annotation_id)

COLS = ["annotation_id", "review_uid", "destination", "language_review", "aspecto", "text_clean", "input_modelo", "label"]
for a in (1, 2, 3):
    ids = list(overlap_ids) + asign[a]
    m = cand[cand["annotation_id"].isin(ids)].copy()
    m["label"] = ""
    m = m.sample(frac=1.0, random_state=SEED + a)[COLS]
    m.to_csv(OUT / f"lote_negativos_anotador_{a}.csv", index=False, encoding="utf-8-sig")
    print(f"  Anotador {a}: {len(m)} items ({len(overlap_ids)} solapados + {len(asign[a])} unicos)")

# Clave maestra (trazabilidad, no para anotadores)
clave = cand[["annotation_id", "review_uid", "destination", "aspecto", "stars", "n_cues_neg", "cues_neg", "calidad_senal"]].copy()
clave["en_overlap"] = clave["annotation_id"].isin(overlap_ids)
clave.to_csv(OUT / "_clave_maestra_negativos.csv", index=False, encoding="utf-8-sig")

# Copiar la guia de anotacion
guia_src = DATA / "anotacion_v2" / "GUIA_ANOTACION.md"
if guia_src.exists():
    (OUT / "GUIA_ANOTACION.md").write_text(guia_src.read_text(encoding="utf-8"), encoding="utf-8")

(OUT / "LEEME.txt").write_text(
    "LOTES DE AMPLIACION DE NEGATIVOS (v2)\n"
    "Estos archivos son CANDIDATOS NUEVOS a anotar, NO el gold existente.\n"
    "Cada anotador llena la columna 'label' (positivo/neutro/negativo) de su\n"
    "archivo lote_negativos_anotador_<n>.csv siguiendo GUIA_ANOTACION.md.\n"
    "Hay items solapados entre los 3 a proposito (para kappa). No comparen entre si.\n"
    "Al terminar, devolver los 3 archivos para consolidar y fusionar al gold.\n",
    encoding="utf-8")
print("Salidas en:", OUT)
print("Total etiquetas a producir:", len(overlap_ids) * 3 + sum(len(asign[a]) for a in (1, 2, 3)))
