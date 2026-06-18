# -*- coding: utf-8 -*-
"""
Valida y limpia los CSV de los anotadores para el notebook 01.

Qué hace (de forma segura e idempotente):
  1. Detecta filas CORRUPTAS (aspecto o label que no son valores válidos) y las reporta.
  2. Elimina filas 100% idénticas (duplicados exactos).
  3. Regenera annotation_id = review_uid + "__" + aspecto (clave única por reseña-aspecto).
  4. Reporta CONFLICTOS (mismo review+aspecto con label distinto) -> requieren decisión humana.
  5. Solo SOBRESCRIBE el archivo si quedó limpio (0 corruptas, 0 conflictos, 0 duplicados de clave).

Uso:
    python scripts/validar_y_limpiar_anotadores.py            # valida/limpia los 3
    python scripts/validar_y_limpiar_anotadores.py erick      # solo el de Erick
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

FILES = {
    "alvaro": "muestra_anotador_1.csv",
    "moises": "muestra_anotador_2.csv",
    "erick": "muestra_anotador_3.csv",
}

VALID_LABELS = {"positivo", "neutro", "negativo"}
VALID_ASPECTS = {
    "atractivos", "costos", "seguridad", "accesibilidad", "limpieza",
    "atencion_servicio", "gastronomia", "alojamiento", "clima", "aforo_multitudes",
}
LABEL_NORMALIZATION = {
    "positiva": "positivo", "pos": "positivo", "+": "positivo",
    "neutral": "neutro", "neutra": "neutro", "nuetro": "neutro",
    "negativa": "negativo", "neg": "negativo", "-": "negativo",
}
ASPECT_NORMALIZATION = {
    "atención_servicio": "atencion_servicio", "atencion/servicio": "atencion_servicio",
    "atencion servicio": "atencion_servicio", "aforo/multitudes": "aforo_multitudes",
    "aforo multitudes": "aforo_multitudes", "multitudes": "aforo_multitudes",
    "atractivo": "atractivos", "atractivos_turisticos": "atractivos",
}


def norm(value, mapping):
    if pd.isna(value):
        return np.nan
    v = str(value).strip().lower()
    if v in ("", "nan", "none", "null"):
        return np.nan
    return mapping.get(v, v)


def procesar(name, path):
    print("=" * 70)
    print(f"ANOTADOR: {name}  ({path.name})")
    print("=" * 70)
    df = pd.read_csv(path, encoding="utf-8-sig")
    n0 = len(df)
    df["annotation_id"] = df["annotation_id"].astype(str).str.strip()
    df["aspecto_n"] = df["aspecto"].apply(lambda v: norm(v, ASPECT_NORMALIZATION))
    df["label_n"] = df["label"].apply(lambda v: norm(v, LABEL_NORMALIZATION))

    # 1) Filas corruptas: aspecto o label no válidos
    bad_asp = ~df["aspecto_n"].isin(VALID_ASPECTS)
    bad_lab = ~df["label_n"].isin(VALID_LABELS)
    corruptas = df[bad_asp | bad_lab]
    if len(corruptas):
        print(f"\n[CORRUPTAS] {len(corruptas)} filas con aspecto/label inválido:")
        for _, r in corruptas.iterrows():
            print(f"  review_uid={r['review_uid']!s:25} aspecto={str(r['aspecto'])[:25]!r:27} "
                  f"label={str(r['label'])[:25]!r}")

    # 2) Quitar filas 100% idénticas
    df_clean = df.drop_duplicates(subset=[c for c in df.columns if c not in ("aspecto_n", "label_n")])
    n1 = len(df_clean)
    if n0 - n1:
        print(f"\n[DUPLICADOS EXACTOS] eliminadas {n0 - n1} filas idénticas.")

    # 3) Regenerar annotation_id (solo para filas con aspecto válido)
    new_id = (df_clean["review_uid"].astype(str).str.strip() + "__"
              + df_clean["aspecto"].astype(str).str.strip())
    df_clean = df_clean.assign(annotation_id=new_id)

    # 4) Conflictos: misma clave (review+aspecto) con label distinto
    key = df_clean["review_uid"].astype(str).str.strip() + "__" + df_clean["aspecto_n"].astype(str)
    conf = df_clean[key.duplicated(keep=False)].sort_values(["review_uid", "aspecto"])
    if len(conf):
        print(f"\n[CONFLICTOS] {len(conf)} filas con misma clave y label distinto (decide manualmente):")
        print(conf[["review_uid", "aspecto", "label"]].to_string(index=False))

    # ¿Quedó limpio?
    limpio = len(corruptas) == 0 and len(conf) == 0
    if limpio:
        salida = df_clean.drop(columns=["aspecto_n", "label_n"])
        salida.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n[OK] Archivo limpio. Guardado: {n1} filas, annotation_id único.")
    else:
        print(f"\n[NO GUARDADO] Corrige las filas reportadas y vuelve a correr el script.")
    print()
    return limpio


def main():
    targets = sys.argv[1:] or list(FILES.keys())
    todos_ok = True
    for name in targets:
        if name not in FILES:
            print(f"Anotador desconocido: {name}. Usa: {list(FILES)}")
            continue
        ok = procesar(name, DATA_DIR / FILES[name])
        todos_ok = todos_ok and ok
    print("=" * 70)
    print("RESULTADO:", "TODOS LIMPIOS — listo para el notebook" if todos_ok
          else "HAY ARCHIVOS POR CORREGIR (ver arriba)")
    print("=" * 70)


if __name__ == "__main__":
    main()
