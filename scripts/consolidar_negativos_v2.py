# -*- coding: utf-8 -*-
# ============================================================
# Consolida los lotes de candidatos negativos (anotacion_negativos_v2) y los
# fusiona con el gold de consenso para producir el gold ampliado v3.
# ------------------------------------------------------------
#   - Items SOLAPADOS (en los 3 anotadores): etiqueta de consenso (mayoria 2/3)
#     + Fleiss kappa como control de fiabilidad del lote nuevo.
#   - Items UNICOS (1 anotador): se toma su etiqueta.
#   - Empates 1/1/1 -> SIN_CONSENSO, no entran al gold.
# Salidas:
#   data/gold_set_v3.csv                         (gold de consenso + nuevos)
#   outputs/reports/kappa_negativos_v2.csv
#   outputs/reports/aporte_negativos_v2.csv      (cuanto crecio cada clase/aspecto)
# No sobrescribe gold_set_consenso_v2.csv ni gold_set_final.csv.
# ============================================================
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd

BASE = Path(__file__).resolve().parent.parent
ANOT = BASE / "data" / "anotacion_negativos_v2"
DATA = BASE / "data"; REP = BASE / "outputs" / "reports"
L = ["negativo", "neutro", "positivo"]
META = ["annotation_id", "review_uid", "destination", "language_review", "aspecto", "text_clean", "input_modelo"]

dfs = {}
for a in (1, 2, 3):
    d = pd.read_csv(ANOT / f"lote_negativos_anotador_{a}.csv", encoding="utf-8-sig")
    d["label"] = d["label"].astype(str).str.lower().str.strip()
    dfs[a] = d

allrows = pd.concat(dfs.values())
meta = allrows[META].drop_duplicates("annotation_id").set_index("annotation_id")
counts = allrows.groupby("annotation_id")["label"].count()
overlap_ids = counts[counts == 3].index
unique_ids = counts[counts == 1].index
print(f"items totales: {meta.shape[0]} | solapados(3): {len(overlap_ids)} | unicos(1): {len(unique_ids)}")

# --- Overlap: consenso + Fleiss kappa ---
lab = pd.DataFrame({f"a{a}": dfs[a].set_index("annotation_id")["label"] for a in (1, 2, 3)})
ov = lab.loc[overlap_ids].dropna()
def maj(r):
    c = Counter([r.a1, r.a2, r.a3]); top, n = c.most_common(1)[0]
    return top if n >= 2 else "SIN_CONSENSO"
def nac(r):
    return Counter([r.a1, r.a2, r.a3]).most_common(1)[0][1]
ov_consenso = ov.apply(maj, axis=1); ov_nac = ov.apply(nac, axis=1)

N, n = len(ov), 3
M = np.zeros((N, len(L)))
for i, (_, r) in enumerate(ov.iterrows()):
    for x in [r.a1, r.a2, r.a3]:
        if x in L: M[i, L.index(x)] += 1
p = M.sum(0) / (N * n); Pi = ((M ** 2).sum(1) - n) / (n * (n - 1))
kappa = (Pi.mean() - (p ** 2).sum()) / (1 - (p ** 2).sum())

def cohen(x, y):
    df = lab.loc[overlap_ids, [x, y]].dropna()
    obs = (df[x] == df[y]).mean()
    pe = sum((df[x] == c).mean() * (df[y] == c).mean() for c in L)
    return (obs - pe) / (1 - pe), obs
kap_rows = []
for (x, y) in [("a1", "a2"), ("a1", "a3"), ("a2", "a3")]:
    k, o = cohen(x, y)
    kap_rows.append({"par": f"{x}_vs_{y}", "n": len(ov), "acuerdo_%": round(o*100, 2), "cohen_kappa": round(k, 4)})
pd.DataFrame([{"n_items_overlap": N, "fleiss_kappa": round(kappa, 4),
               "acuerdo_3de3": int((ov_nac == 3).sum()), "acuerdo_2de3": int((ov_nac == 2).sum()),
               "sin_consenso": int((ov_consenso == "SIN_CONSENSO").sum())}] + kap_rows
             ).to_csv(REP / "kappa_negativos_v2.csv", index=False, encoding="utf-8-sig")
print(f"Fleiss kappa (lote negativos, {N} solapados): {kappa:.4f}")

# --- Etiqueta final por item ---
final = {}
for i in overlap_ids:
    if ov_consenso[i] != "SIN_CONSENSO":
        final[i] = ov_consenso[i]
for i in unique_ids:
    v = lab.loc[i].dropna()
    if len(v): final[i] = v.iloc[0]
nuevos = meta.loc[list(final.keys())].copy()
nuevos["label"] = pd.Series(final)
nuevos["origen"] = "ampliacion_negativos_v2"
nuevos = nuevos.reset_index()

print("Aporte de la ampliacion (etiquetas finales):", nuevos["label"].value_counts().to_dict())

# --- Fusion con gold de consenso ---
gold = pd.read_csv(DATA / "gold_set_consenso_v2.csv", encoding="utf-8-sig")
gold["origen"] = "consenso_v2"
keep = [c for c in ["annotation_id", "review_uid", "destination", "language_review", "aspecto", "text_clean", "input_modelo", "label", "origen"]]
g3 = pd.concat([gold[ [c for c in keep if c in gold.columns] ],
                nuevos[[c for c in keep if c in nuevos.columns]]], ignore_index=True)
g3 = g3.drop_duplicates("annotation_id", keep="first")
g3.to_csv(DATA / "gold_set_v3.csv", index=False, encoding="utf-8-sig")

# --- Reporte de aporte por aspecto/polaridad ---
piv = g3.pivot_table(index="aspecto", columns="label", values="annotation_id", aggfunc="count", fill_value=0)
for c in L:
    if c not in piv: piv[c] = 0
piv = piv[L]; piv["total"] = piv.sum(axis=1)
piv.to_csv(REP / "aporte_negativos_v2.csv", encoding="utf-8-sig")

print(f"\nGOLD v3: {len(g3)} items -> {DATA/'gold_set_v3.csv'}")
print("Distribucion global:", g3["label"].value_counts().to_dict())
print("\nPor aspecto x label (gold v3):")
print(piv.to_string())
