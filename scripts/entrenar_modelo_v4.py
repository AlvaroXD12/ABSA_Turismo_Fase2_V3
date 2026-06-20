# -*- coding: utf-8 -*-
# Entrena UN modelo (en su propio proceso -> GPU fresca, sin OOM por acumulación).
# Uso:
#   python scripts/entrenar_modelo_v4.py --model xlmr   # candidato principal (hace la búsqueda de HP)
#   python scripts/entrenar_modelo_v4.py --model bert   # base oficial (reutiliza el HP elegido)
# El notebook 03 carga los artefactos que esto genera y produce el reporte + matriz.
import argparse, time
import absa_common as ac

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["xlmr", "bert"])
    args = ap.parse_args()
    print(f"DEVICE: {ac.DEVICE} | AMP: {ac.USE_AMP} | modelo: {args.model}", flush=True)
    if ac.DEVICE.type != "cuda":
        print("⚠️  ADVERTENCIA: no se detectó GPU (CUDA). Entrenará en CPU y será MUY lento.", flush=True)
    train, val, test = ac.load_splits()
    print(f"train={len(train)} val={len(val)} test={len(test)}", flush=True)
    t0 = time.time()
    neg_boost, focal_gamma = ac.get_hp(train, val, test)   # cacheado en HP_FILE; XLM-R lo crea, BERT lo reutiliza
    print(f"HP (por validación): NEG_BOOST={neg_boost}, FOCAL_GAMMA={focal_gamma}", flush=True)
    ac.run_modelo(args.model, train, val, test, neg_boost, focal_gamma)
    print(f"\n✅ {args.model} completado en {(time.time()-t0)/60:.1f} min. Artefactos en outputs/.", flush=True)

if __name__ == "__main__":
    main()
