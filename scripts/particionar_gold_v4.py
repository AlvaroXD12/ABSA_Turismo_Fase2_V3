# -*- coding: utf-8 -*-
# ============================================================
# Paso 6 - Particion documentada del gold set v4 (3562 items).
# ------------------------------------------------------------
# Requisitos:
#  - Particionar por review_uid (grupo), NO por fila -> sin fuga train/val/test.
#  - 70/15/15 con estratificacion aproximada por polaridad (ajustes minimos
#    obligados por el agrupamiento).
#  - Trazabilidad completa por fila + split asignado.
#  - Auditoria por polaridad, aspecto, destino, idioma.
#  - Verificacion explicita de no-fuga (0 review_uid compartidos).
#  - NO reutiliza el test viejo ni ajusta el split para inflar resultados.
# Salidas (en data/ y outputs/reports/):
#   train_gold_v4.csv, val_gold_v4.csv, test_gold_v4.csv
#   split_report_gold_v4.md  y  split_report_gold_v4.json
# ============================================================
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"; REP = BASE / "outputs" / "reports"
REP.mkdir(parents=True, exist_ok=True)
SEED = 42
N_SPLITS = 20            # 20 folds ~5% c/u -> test=3 folds(~15%), val=3(~15%), train=14(~70%)
TEST_FOLDS, VAL_FOLDS = {0, 1, 2}, {3, 4, 5}
LABELS = ["negativo", "neutro", "positivo"]

g = pd.read_csv(DATA / "gold_set_v4.csv", encoding="utf-8-sig")
g["label"] = g["label"].astype(str).str.lower().str.strip()
g = g[g["label"].isin(LABELS)].reset_index(drop=True)
if "annotation_id" not in g.columns:
    g["annotation_id"] = g["review_uid"].astype(str) + "__" + g["aspecto"].astype(str)

# --- Particion estratificada por grupo (review_uid) ---
skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
fold_of_row = np.full(len(g), -1, dtype=int)
for fold_idx, (_, test_idx) in enumerate(skf.split(g, g["label"], groups=g["review_uid"])):
    fold_of_row[test_idx] = fold_idx
g["_fold"] = fold_of_row


def split_de_fold(f):
    if f in TEST_FOLDS: return "test"
    if f in VAL_FOLDS: return "val"
    return "train"


g["split"] = g["_fold"].map(split_de_fold)

# --- Verificacion de no-fuga por review_uid ---
sets = {s: set(g.loc[g.split == s, "review_uid"]) for s in ["train", "val", "test"]}
fuga = {
    "train_val": sorted(sets["train"] & sets["val"]),
    "train_test": sorted(sets["train"] & sets["test"]),
    "val_test": sorted(sets["val"] & sets["test"]),
}
hay_fuga = any(len(v) for v in fuga.values())
assert not hay_fuga, f"FUGA detectada: {fuga}"

# --- Guardar archivos por split (trazabilidad completa) ---
COLS = [c for c in ["annotation_id", "review_uid", "destination", "language_review",
                    "aspecto", "text_clean", "input_modelo", "label", "origen", "split"] if c in g.columns]
for s, fname in [("train", "train_gold_v4.csv"), ("val", "val_gold_v4.csv"), ("test", "test_gold_v4.csv")]:
    g.loc[g.split == s, COLS].to_csv(DATA / fname, index=False, encoding="utf-8-sig")


# --- Auditoria ---
def tabla(col):
    t = g.pivot_table(index=col, columns="split", values="annotation_id", aggfunc="count", fill_value=0)
    for s in ["train", "val", "test"]:
        if s not in t: t[s] = 0
    return t[["train", "val", "test"]]


total = len(g)
resumen = {}
for s in ["train", "val", "test"]:
    sub = g[g.split == s]
    resumen[s] = {
        "items": int(len(sub)),
        "pct_items": round(len(sub) / total * 100, 2),
        "review_uids": int(sub["review_uid"].nunique()),
        "polaridad": sub["label"].value_counts().reindex(LABELS, fill_value=0).astype(int).to_dict(),
        "polaridad_pct": (sub["label"].value_counts(normalize=True).reindex(LABELS, fill_value=0) * 100).round(2).to_dict(),
    }

aud_aspecto = tabla("aspecto")
aud_destino = tabla("destination") if "destination" in g.columns else pd.DataFrame()
aud_idioma = tabla("language_review") if "language_review" in g.columns else pd.DataFrame()
aud_polaridad = tabla("label")

reporte = {
    "fuente": "data/gold_set_v4.csv",
    "total_items": total,
    "total_review_uids": int(g["review_uid"].nunique()),
    "metodo": "StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42); test=folds{0,1,2}, val=folds{3,4,5}, train=resto",
    "objetivo_proporcion": "70/15/15",
    "proporcion_real": {s: resumen[s]["pct_items"] for s in resumen},
    "no_fuga_review_uid": (not hay_fuga),
    "solapamientos": {k: len(v) for k, v in fuga.items()},
    "por_split": resumen,
}
with open(REP / "split_report_gold_v4.json", "w", encoding="utf-8") as f:
    json.dump(reporte, f, ensure_ascii=False, indent=2)


# --- Reporte markdown ---
def md_tabla(t, titulo):
    out = [f"### {titulo}", "", "| " + titulo.split()[-1] + " | train | val | test |", "|---|---|---|---|"]
    for idx, row in t.iterrows():
        out.append(f"| {idx} | {int(row['train'])} | {int(row['val'])} | {int(row['test'])} |")
    return "\n".join(out) + "\n"


lines = []
lines.append("# Reporte de partición — gold set v4\n")
lines.append(f"- **Fuente:** `data/gold_set_v4.csv` — {total} ítems, {g['review_uid'].nunique()} reseñas únicas.")
lines.append(f"- **Método:** StratifiedGroupKFold (n_splits=20, shuffle, seed=42), agrupado por `review_uid`.")
lines.append(f"  test = folds {{0,1,2}}, val = folds {{3,4,5}}, train = resto. Objetivo 70/15/15.")
lines.append(f"- **No-fuga por `review_uid`:** {'✅ SIN solapamiento' if not hay_fuga else '❌ FUGA'} "
             f"(train∩val={len(fuga['train_val'])}, train∩test={len(fuga['train_test'])}, val∩test={len(fuga['val_test'])}).\n")
lines.append("## Tamaño por split\n")
lines.append("| split | ítems | % | reseñas únicas |")
lines.append("|---|---|---|---|")
for s in ["train", "val", "test"]:
    r = resumen[s]
    lines.append(f"| {s} | {r['items']} | {r['pct_items']} | {r['review_uids']} |")
lines.append("")
lines.append("## Polaridad por split (conteo / %)\n")
lines.append("| split | negativo | neutro | positivo |")
lines.append("|---|---|---|---|")
for s in ["train", "val", "test"]:
    p, pp = resumen[s]["polaridad"], resumen[s]["polaridad_pct"]
    lines.append(f"| {s} | {p['negativo']} ({pp['negativo']}%) | {p['neutro']} ({pp['neutro']}%) | {p['positivo']} ({pp['positivo']}%) |")
lines.append("")
lines.append(md_tabla(aud_aspecto, "Aspecto"))
if not aud_idioma.empty: lines.append(md_tabla(aud_idioma, "Idioma"))
if not aud_destino.empty: lines.append(md_tabla(aud_destino, "Destino"))
lines.append("## Notas\n")
lines.append("- Partición por grupo `review_uid`: ninguna reseña aparece en más de un split (sin fuga).")
lines.append("- El test es NUEVO (derivado del gold v4), no se reutilizó el test anterior.")
lines.append("- La estratificación es aproximada: el agrupamiento por reseña impone pequeños ajustes sobre el 70/15/15 exacto.")
(REP / "split_report_gold_v4.md").write_text("\n".join(lines), encoding="utf-8")

print("=== PARTICIÓN gold v4 ===")
print(f"total {total} | train {resumen['train']['items']} ({resumen['train']['pct_items']}%) "
      f"| val {resumen['val']['items']} ({resumen['val']['pct_items']}%) "
      f"| test {resumen['test']['items']} ({resumen['test']['pct_items']}%)")
print("No-fuga review_uid:", "OK" if not hay_fuga else "FUGA")
for s in ["train", "val", "test"]:
    print(f"  {s}: polaridad {resumen[s]['polaridad']}")
print("Archivos:", DATA / "train_gold_v4.csv", DATA / "val_gold_v4.csv", DATA / "test_gold_v4.csv")
print("Reporte:", REP / "split_report_gold_v4.md")
