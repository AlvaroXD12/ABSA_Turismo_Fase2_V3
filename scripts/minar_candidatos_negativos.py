# -*- coding: utf-8 -*-
# ============================================================
# Minado de CANDIDATOS negativos para ampliar el gold set
# ------------------------------------------------------------
# NO etiqueta gold: produce una cola de candidatos de ALTA probabilidad de ser
# negativos para que un ANOTADOR HUMANO los verifique. La señal principal es la
# estrella baja (1-2), que es independiente del modelo ABSA y la más confiable
# para negativos; se rankea por pistas léxicas negativas (es/en) dentro de cada
# aspecto. Se excluyen pares (review_uid, aspecto) ya presentes en el gold set y
# se balancea por destino para no sobre-representar Machu Picchu.
#
# Entradas:
#   outputs/predictions/tourism_reviews_clean_absa_ready.csv  (reseña x aspecto)
#   data/tourism_reviews_clean.csv                            (estrellas)
#   data/gold_set_final.csv                                   (para excluir)
# Salidas:
#   data/candidatos_negativos_para_anotacion.csv  (cola de anotación)
#   outputs/reports/diagnostico_gold_set.csv      (déficit por aspecto/polaridad)
#   outputs/reports/disponibilidad_candidatos.csv (cuántos hay vs objetivo)
# ============================================================
from pathlib import Path
import re, unicodedata
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA, REP = BASE / "data", BASE / "outputs" / "reports"
REP.mkdir(parents=True, exist_ok=True)

ASPECTOS_PRIORITARIOS = ["costos", "seguridad", "accesibilidad", "limpieza",
                         "atencion_servicio", "aforo_multitudes", "alojamiento"]
# Objetivo de negativos por aspecto prioritario (para fortalecer la señal).
OBJETIVO_NEG_POR_ASPECTO = 120
# Se minan más candidatos que el déficit, porque no todos serán negativos reales
# tras la verificación humana (factor de sobre-muestreo).
FACTOR_SOBRE_MUESTREO = 2.0

# Pistas léxicas negativas bilingües (solo para RANKEAR, no para etiquetar).
CUES_NEG = [
    "caro", "carisimo", "estafa", "robo", "cobran", "sobreprecio", "no vale",
    "sucio", "sucia", "mugre", "basura", "asqueroso", "hediondo", "abandonado",
    "inseguro", "peligroso", "robaron", "asalto", "ladrones", "cuidado",
    "lleno", "saturado", "tumulto", "aglomeracion", "colas", "esperar horas",
    "pesimo", "pesima", "terrible", "horrible", "malisimo", "decepcion",
    "decepcionante", "maltrato", "grosero", "groseros", "mala atencion",
    "no recomiendo", "una perdida", "perdida de tiempo", "evitar", "nunca",
    "demora", "lento", "desorganizado", "no funciona", "cerrado", "incompleto",
    "expensive", "rip off", "overpriced", "scam", "dirty", "filthy", "trash",
    "unsafe", "dangerous", "robbed", "theft", "crowded", "overcrowded", "queue",
    "long wait", "awful", "terrible", "horrible", "worst", "disappointing",
    "rude", "poor service", "not worth", "waste of time", "avoid", "broken",
    "closed", "slow", "disorganized", "mess",
]


def norm(t):
    t = str(t).lower()
    t = unicodedata.normalize("NFKD", t)
    return "".join(c for c in t if not unicodedata.combining(c))


CUES_NORM = [norm(c) for c in CUES_NEG]


def contar_cues(texto):
    t = norm(texto)
    hits = [c for c in CUES_NORM if c in t]
    return len(hits), ";".join(sorted(set(hits))[:6])


# --- Carga ---
absa = pd.read_csv(BASE / "outputs/predictions/tourism_reviews_clean_absa_ready.csv", encoding="utf-8-sig")
clean = pd.read_csv(DATA / "tourism_reviews_clean.csv", encoding="utf-8-sig",
                    usecols=["review_uid", "stars", "sentiment_by_stars"])
gold = pd.read_csv(DATA / "gold_set_final.csv", encoding="utf-8-sig")
gold["label"] = gold["label"].astype(str).str.lower().str.strip()

# --- Diagnóstico (déficit por aspecto/polaridad) ---
piv = gold.pivot_table(index="aspecto", columns="label", values="review_uid", aggfunc="count", fill_value=0)
for c in ["negativo", "neutro", "positivo"]:
    if c not in piv: piv[c] = 0
piv = piv[["negativo", "neutro", "positivo"]]
piv["total"] = piv.sum(axis=1)
piv["objetivo_neg"] = piv.index.to_series().apply(lambda a: OBJETIVO_NEG_POR_ASPECTO if a in ASPECTOS_PRIORITARIOS else None)
piv["deficit_neg"] = (piv["objetivo_neg"] - piv["negativo"]).clip(lower=0)
piv.to_csv(REP / "diagnostico_gold_set.csv", encoding="utf-8-sig")

# --- Pares ya en gold (para excluir) ---
gold_pairs = set(zip(gold["review_uid"], gold["aspecto"]))

# --- Construir pool de candidatos ---
absa = absa.merge(clean, on="review_uid", how="left")
absa["stars"] = pd.to_numeric(absa["stars"], errors="coerce")

filas, disponibilidad = [], []
for asp in ASPECTOS_PRIORITARIOS:
    actual_neg = int(piv.loc[asp, "negativo"]) if asp in piv.index else 0
    deficit = max(0, OBJETIVO_NEG_POR_ASPECTO - actual_neg)
    objetivo_minar = int(round(deficit * FACTOR_SOBRE_MUESTREO))

    todas = absa[absa["aspecto"] == asp].copy()
    n_total_corpus = len(todas)
    # excluir los que ya están anotados para ese aspecto
    todas = todas[~todas.apply(lambda r: (r["review_uid"], asp) in gold_pairs, axis=1)]
    todas = todas.drop_duplicates(subset=["review_uid", "aspecto"])

    cues = todas["text_clean"].apply(contar_cues)
    todas["n_cues_neg"] = [c[0] for c in cues]
    todas["cues_neg"] = [c[1] for c in cues]
    # Candidato negativo = tiene pistas léxicas negativas O estrella baja (1-2).
    # Esto recupera negativos de aspecto embebidos en reseñas de 4-5 estrellas.
    cand = todas[(todas["n_cues_neg"] >= 1) | (todas["stars"].isin([1, 2]))].copy()
    # Calidad de señal: ALTA = estrella baja (1-2) o >=2 pistas léxicas (negativo
    # más probable); MEDIA = resto (1 pista en reseña de 3-5 estrellas, más ruido).
    cand["calidad_senal"] = (
        (cand["stars"].isin([1, 2])) | (cand["n_cues_neg"] >= 2)
    ).map({True: "alta", False: "media"})
    cand["_orden_calidad"] = (cand["calidad_senal"] == "alta").astype(int)
    cand["prioridad_estrella"] = cand["stars"].map({1: 3, 2: 2, 3: 1}).fillna(0)
    # priorizar: calidad alta primero, luego más pistas, luego estrella más baja
    cand = cand.sort_values(["_orden_calidad", "n_cues_neg", "prioridad_estrella"], ascending=False)

    disponibles = len(cand)
    # balance por destino: round-robin hasta el objetivo
    seleccion = []
    if objetivo_minar > 0 and disponibles > 0:
        grupos = {d: g.to_dict("records") for d, g in cand.groupby("destination")}
        punteros = {d: 0 for d in grupos}
        while len(seleccion) < objetivo_minar and any(punteros[d] < len(grupos[d]) for d in grupos):
            for d in list(grupos.keys()):
                if punteros[d] < len(grupos[d]):
                    seleccion.append(grupos[d][punteros[d]])
                    punteros[d] += 1
                    if len(seleccion) >= objetivo_minar:
                        break

    for r in seleccion:
        filas.append({
            "review_uid": r["review_uid"], "destination": r["destination"],
            "language_review": r.get("language_review", ""), "aspecto": asp,
            "stars": r["stars"], "sentiment_by_stars": r.get("sentiment_by_stars", ""),
            "n_cues_neg": r["n_cues_neg"], "cues_neg": r["cues_neg"],
            "calidad_senal": r["calidad_senal"],
            "text_clean": r["text_clean"], "input_modelo": r.get("input_modelo", ""),
            "sugerencia_heuristica": "negativo (heurística por estrellas 1-2; REQUIERE verificación humana)",
            "label": "",  # <-- el anotador llena positivo/neutro/negativo
        })
    disponibilidad.append({
        "aspecto": asp, "menciones_corpus": n_total_corpus, "neg_actual_gold": actual_neg,
        "objetivo_neg": OBJETIVO_NEG_POR_ASPECTO, "deficit_neg": deficit,
        "candidatos_a_minar": objetivo_minar,
        "candidatos_disponibles_corpus": disponibles, "candidatos_incluidos": len(seleccion),
        "suficiente": disponibles >= deficit,
    })

cands = pd.DataFrame(filas)
cands.to_csv(DATA / "candidatos_negativos_para_anotacion.csv", index=False, encoding="utf-8-sig")
disp = pd.DataFrame(disponibilidad)
disp.to_csv(REP / "disponibilidad_candidatos.csv", index=False, encoding="utf-8-sig")

print("=== DISPONIBILIDAD DE CANDIDATOS NEGATIVOS (estrellas 1-2, no anotados) ===")
print(disp.to_string(index=False))
print(f"\nTotal candidatos en la cola de anotación: {len(cands)}")
print("Archivo:", DATA / "candidatos_negativos_para_anotacion.csv")
print("Diagnóstico:", REP / "diagnostico_gold_set.csv")
