# -*- coding: utf-8 -*-
# Valida que el notebook v4 corre de principio a fin SIN GPU, datos diminutos,
# 1 semilla, 1 epoca, sin XLM-R, corpus recortado. Detecta errores antes de la PC potente.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["MPLBACKEND"] = "Agg"
import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "03_entrenamiento_absa_bert_textcnn_gold_v4.ipynb"
code_cells = [''.join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
ns = {"display": lambda *a, **k: None}
shrunk = False
for i, src in enumerate(code_cells):
    print(f"--- celda {i} ---")
    exec(src, ns)
    if "SEEDS" in ns and ns.get("EPOCHS", 99) != 1:
        ns.update(EPOCHS=1, SEEDS=[42], BATCH=4, INCLUIR_XLMR=False, MAX_CORPUS_INFER=40, RUN_TRAINING=True)
        print("  [override] mini config")
    if (not shrunk) and all(k in ns for k in ("train", "val", "test")):
        mini = lambda d: d.groupby("label", group_keys=False).head(8).reset_index(drop=True)
        ns["train"], ns["val"], ns["test"] = mini(ns["train"]), mini(ns["val"]), mini(ns["test"])
        shrunk = True; print(f"  [override] splits {len(ns['train'])}/{len(ns['val'])}/{len(ns['test'])}")
print("\nVALIDACION OK: notebook v4 corre end-to-end. Veredicto:", ns.get("VEREDICTO"))
