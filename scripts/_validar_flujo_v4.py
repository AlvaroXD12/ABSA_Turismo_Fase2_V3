# -*- coding: utf-8 -*-
# Valida el flujo completo opción B en CPU con datos mini:
#   1) entrena xlmr y bert (mini) con absa_common -> genera artefactos
#   2) ejecuta el notebook-reporte que los carga + matriz + gráficos
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["MPLBACKEND"] = "Agg"
import sys, json
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE / "scripts"))
import absa_common as ac

# --- config mini ---
ac.SEEDS = [42]; ac.EPOCHS = 1; ac.SEARCH_EPOCHS = 1; ac.NEG_BOOST_GRID = [1.2]; ac.FOCAL_GRID = [2.0]
ac.MAX_LEN = 128; ac.BATCH = 4; ac.USE_GRADIENT_CHECKPOINTING = False
ac.MODELOS = {"xlmr": "bert-base-multilingual-cased", "bert": "bert-base-multilingual-cased"}  # ambos mBERT (rápido)

_orig = ac.load_splits
def mini_splits():
    tr, va, te = _orig(); m = lambda d: d.groupby("label", group_keys=False).head(8).reset_index(drop=True)
    return m(tr), m(va), m(te)
ac.load_splits = mini_splits

print("== Entrenando mini xlmr + bert ==")
train, val, test = ac.load_splits()
nb, fg = ac.get_hp(train, val, test)
ac.run_modelo("xlmr", train, val, test, nb, fg)
ac.run_modelo("bert", train, val, test, nb, fg)

print("\n== Ejecutando notebook-reporte ==")
NB = BASE / "notebooks" / "03_reporte_absa_xlmr_bert_gold_v4.ipynb"
code_cells = [''.join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
ns = {"display": lambda *a, **k: None}
for i, src in enumerate(code_cells):
    print(f"--- celda {i} ---")
    exec(src, ns)
    if ns.get("MAX_CORPUS_INFER", "x") is None:
        ns["MAX_CORPUS_INFER"] = 40
print("\nVALIDACION OK (flujo opción B). Modelo matriz:", ns.get("MODELO_FINAL"))
