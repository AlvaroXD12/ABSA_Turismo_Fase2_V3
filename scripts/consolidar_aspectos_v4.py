# -*- coding: utf-8 -*-
# Consolida los lotes de refuerzo por aspecto (clima/aforo/limpieza) y los fusiona
# con gold_set_v3 -> gold_set_v4. Etiqueta de consenso (mayoria 2/3) + Fleiss kappa.
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA, REP = BASE / "data", BASE / "outputs" / "reports"
L = ["negativo", "neutro", "positivo"]
META = ["review_uid", "destination", "language_review", "aspecto", "text_clean", "input_modelo"]

dfs = {a: pd.read_csv(DATA / f"candidatos_aspectos_anotador_{a}.csv", encoding="utf-8-sig") for a in (1, 2, 3)}
for a in dfs: dfs[a]["label"] = dfs[a]["label"].str.lower().str.strip()
base = dfs[1].copy()
base["annotation_id"] = base["review_uid"].astype(str) + "__" + base["aspecto"].astype(str)
lab = pd.DataFrame({f"a{a}": dfs[a].set_index("review_uid")["label"].reindex(base["review_uid"]).values for a in (1, 2, 3)})

def maj(r):
    c = Counter([r.a1, r.a2, r.a3]); t, n = c.most_common(1)[0]; return t if n >= 2 else "SIN_CONSENSO"
base["label"] = lab.apply(maj, axis=1)

# Fleiss kappa
N = len(lab); M = np.zeros((N, 3))
for i, r in lab.iterrows():
    for x in [r.a1, r.a2, r.a3]:
        if x in L: M[i, L.index(x)] += 1
p = M.sum(0) / (N * 3); Pi = ((M ** 2).sum(1) - 3) / 6
kappa = (Pi.mean() - (p ** 2).sum()) / (1 - (p ** 2).sum())

nuevos = base[base["label"].isin(L)][META + ["annotation_id", "label"]].copy()
nuevos["origen"] = "refuerzo_aspectos_v4"

gold = pd.read_csv(DATA / "gold_set_v3.csv", encoding="utf-8-sig")
if "annotation_id" not in gold.columns:
    gold["annotation_id"] = gold["review_uid"].astype(str) + "__" + gold["aspecto"].astype(str)
gold["label"] = gold["label"].str.lower()
if "origen" not in gold.columns: gold["origen"] = "gold_v3"
keep = ["annotation_id", "review_uid", "destination", "language_review", "aspecto", "text_clean", "input_modelo", "label", "origen"]
g4 = pd.concat([gold[[c for c in keep if c in gold.columns]], nuevos[[c for c in keep if c in nuevos.columns]]],
               ignore_index=True).drop_duplicates("annotation_id", keep="first")
g4.to_csv(DATA / "gold_set_v4.csv", index=False, encoding="utf-8-sig")

print(f"Fleiss kappa (refuerzo aspectos): {kappa:.4f} | items anotados: {N} | nuevos al gold: {len(nuevos)}")
print(f"GOLD v4: {len(g4)} (v3 era {len(gold)}) -> data/gold_set_v4.csv")
print("\n=== Distribucion por aspecto: v3 -> v4 (foco) ===")
foco = ["clima", "aforo_multitudes", "limpieza", "alojamiento"]
for a in foco:
    v3 = gold[gold.aspecto == a]["label"].value_counts()
    v4 = g4[g4.aspecto == a]["label"].value_counts()
    f = lambda s: f"{s.get('negativo',0)}/{s.get('neutro',0)}/{s.get('positivo',0)}"
    print(f"  {a:18s} v3 {f(v3):14s} -> v4 {f(v4):14s} (neg/neu/pos)")
piv = g4.pivot_table(index="aspecto", columns="label", values="annotation_id", aggfunc="count", fill_value=0)
piv.to_csv(REP / "distribucion_gold_v4.csv", encoding="utf-8-sig")
