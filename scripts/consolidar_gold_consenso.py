# -*- coding: utf-8 -*-
# ============================================================
# Consolida el gold mejorado a partir de la re-anotacion triple (2869 items).
# ------------------------------------------------------------
# Los 3 anotadores etiquetaron los MISMOS items (solapamiento total), asi que:
#   - se calcula Fleiss kappa global (calidad del acuerdo),
#   - la etiqueta de consenso = mayoria (>=2/3); los empates 1/1/1 quedan SIN
#     consenso y se separan para adjudicacion (NO entran al gold).
# Salidas:
#   data/gold_set_consenso_v2.csv            (gold mejorado, etiqueta de consenso)
#   outputs/reports/kappa_consenso_v2.csv    (kappa global)
#   outputs/reports/desacuerdos_consenso_v2.csv (items sin acuerdo 3/3)
#   outputs/reports/sin_consenso_v2.csv      (empates a adjudicar)
# NO sobrescribe gold_set_final.csv (se conserva por trazabilidad).
# ============================================================
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd

BASE = Path(__file__).resolve().parent.parent
ANOT = BASE / "data" / "anotacion_v2"
DATA = BASE / "data"; REP = BASE / "outputs" / "reports"
L = ["negativo", "neutro", "positivo"]

dfs = []
for a in (1, 2, 3):
    d = pd.read_csv(ANOT / f"muestra_anotador_{a}.csv", encoding="utf-8-sig")
    d["label"] = d["label"].astype(str).str.lower().str.strip()
    dfs.append(d)

meta_cols = ["annotation_id", "review_uid", "destination", "language_review", "aspecto", "text_clean", "input_modelo"]
meta = dfs[0][meta_cols].drop_duplicates("annotation_id").set_index("annotation_id")

lab = pd.DataFrame({f"a{a}": dfs[a-1].set_index("annotation_id")["label"] for a in (1, 2, 3)})
lab = lab.dropna()

def consenso(r):
    c = Counter([r.a1, r.a2, r.a3]); top, n = c.most_common(1)[0]
    return top if n >= 2 else "SIN_CONSENSO"
def n_acuerdo(r):
    return Counter([r.a1, r.a2, r.a3]).most_common(1)[0][1]

lab["consenso"] = lab.apply(consenso, axis=1)
lab["n_acuerdo"] = lab.apply(n_acuerdo, axis=1)

# Fleiss kappa
N, n = len(lab), 3
M = np.zeros((N, len(L)))
for i, (_, r) in enumerate(lab.iterrows()):
    for x in [r.a1, r.a2, r.a3]:
        if x in L: M[i, L.index(x)] += 1
p = M.sum(0) / (N * n)
Pi = ((M ** 2).sum(1) - n) / (n * (n - 1))
kappa = (Pi.mean() - (p ** 2).sum()) / (1 - (p ** 2).sum())

# Gold de consenso (excluye SIN_CONSENSO)
ok = lab[lab["consenso"] != "SIN_CONSENSO"].copy()
gold = meta.loc[ok.index].copy()
gold["label"] = ok["consenso"]
gold["n_acuerdo"] = ok["n_acuerdo"]
gold = gold.reset_index()
gold.to_csv(DATA / "gold_set_consenso_v2.csv", index=False, encoding="utf-8-sig")

# Reportes
pd.DataFrame([{"n_items": N, "n_anotadores": 3, "fleiss_kappa": round(kappa, 4),
               "acuerdo_3de3": int((lab.n_acuerdo == 3).sum()),
               "acuerdo_2de3": int((lab.n_acuerdo == 2).sum()),
               "sin_consenso": int((lab.consenso == "SIN_CONSENSO").sum())}]
             ).to_csv(REP / "kappa_consenso_v2.csv", index=False, encoding="utf-8-sig")
des = lab[lab.n_acuerdo < 3].join(meta[["aspecto", "text_clean"]])
des.reset_index().to_csv(REP / "desacuerdos_consenso_v2.csv", index=False, encoding="utf-8-sig")
lab[lab.consenso == "SIN_CONSENSO"].join(meta[["aspecto", "text_clean"]]).reset_index().to_csv(
    REP / "sin_consenso_v2.csv", index=False, encoding="utf-8-sig")

print(f"Fleiss kappa: {kappa:.4f}  | items {N}")
print("Gold de consenso:", len(gold), "->", DATA / "gold_set_consenso_v2.csv")
print("Distribución consenso:", gold["label"].value_counts().to_dict())
print("Por aspecto x label:")
print(gold.pivot_table(index="aspecto", columns="label", values="annotation_id", aggfunc="count", fill_value=0).to_string())
