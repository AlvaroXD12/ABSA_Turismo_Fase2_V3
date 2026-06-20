# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["MPLBACKEND"] = "Agg"
import json
from pathlib import Path
NB = Path(__file__).resolve().parent.parent / "notebooks" / "03_entrenamiento_absa_xlmr_bert_gold_v4.ipynb"
code_cells = [''.join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
ns = {"display": lambda *a, **k: None}; shrunk = False
for i, src in enumerate(code_cells):
    print(f"--- celda {i} ---"); exec(src, ns)
    if "SEEDS" in ns and ns.get("EPOCHS", 99) != 1:
        ns.update(EPOCHS=1, SEARCH_EPOCHS=1, SEEDS=[42], BATCH=4, NEG_BOOST_GRID=[1.2], FOCAL_GRID=[2.0],
                  MAX_CORPUS_INFER=40, RUN_TRAINING=True, MODEL_XLMR=ns["MODEL_BERT"])
        print("  [override] mini config (XLMR=BERT para rapidez)")
    if (not shrunk) and all(k in ns for k in ("train", "val", "test")):
        mini = lambda d: d.groupby("label", group_keys=False).head(8).reset_index(drop=True)
        ns["train"], ns["val"], ns["test"] = mini(ns["train"]), mini(ns["val"]), mini(ns["test"]); shrunk = True
        print(f"  [override] splits {len(ns['train'])}/{len(ns['val'])}/{len(ns['test'])}")
print("\nVALIDACION OK. Veredicto:", ns.get("VEREDICTO"), "| modelo matriz:", ns.get("MODELO_FINAL"))
