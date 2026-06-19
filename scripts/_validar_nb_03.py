# -*- coding: utf-8 -*-
# Valida que el notebook 03 corre de principio a fin SIN GPU y con datos diminutos
# (1 epoca, 1 semilla, 24 filas/split). Detecta errores de API/logica antes de que
# el notebook se ejecute en otra PC. NO usa la GPU (evita chocar con el entrenamiento).
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # fuerza CPU
os.environ["MPLBACKEND"] = "Agg"          # sin ventana grafica
import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "03_entrenamiento_absa_bert_textcnn_gold_v3.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
code_cells = [ ''.join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" ]

ns = {"display": lambda *a, **k: None}   # display() no existe fuera de IPython
shrunk = False
for i, src in enumerate(code_cells):
    print(f"\n----- ejecutando celda de codigo #{i} -----")
    exec(src, ns)
    # Tras la celda de config, reducir a lo minimo
    if "SEEDS" in ns and ns.get("EPOCHS", 0) != 1:
        ns["EPOCHS"] = 1; ns["SEEDS"] = [42]; ns["BATCH"] = 4
        print("  [override] EPOCHS=1, SEEDS=[42], BATCH=4")
    # Tras cargar los splits, submuestrear a 24 filas equilibradas
    if (not shrunk) and all(k in ns for k in ("train", "val", "test")):
        import pandas as pd
        def mini(df):
            return (df.groupby("label", group_keys=False).head(8)).reset_index(drop=True)
        ns["train"], ns["val"], ns["test"] = mini(ns["train"]), mini(ns["val"]), mini(ns["test"])
        shrunk = True
        print(f"  [override] splits reducidos a {len(ns['train'])}/{len(ns['val'])}/{len(ns['test'])} filas")
    # Tras cargar el corpus de inferencia, reducirlo (si no, son 21040 filas en CPU)
    if "corpus" in ns and len(ns["corpus"]) > 60:
        ns["corpus"] = ns["corpus"].head(40).reset_index(drop=True)
        print(f"  [override] corpus de inferencia reducido a {len(ns['corpus'])} filas")

print("\n==================================================")
print("VALIDACION OK: el notebook corre de principio a fin (CPU, datos mini).")
print("Veredicto de prueba:", ns.get("VEREDICTO"))
