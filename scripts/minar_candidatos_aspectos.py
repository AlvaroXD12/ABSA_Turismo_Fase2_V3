# -*- coding: utf-8 -*-
# ============================================================
# Minado de CANDIDATOS por aspecto x polaridad para reforzar los aspectos debiles
# (clima, aforo_multitudes, limpieza). NO etiqueta: produce candidatos de alta
# probabilidad de cada polaridad para que un HUMANO los verifique, priorizando las
# clases flacas de cada aspecto. Excluye lo ya anotado en gold_set_v3.
#
# Senales (solo para PRIORIZAR, no etiquetar):
#   negativo  -> estrellas 1-2 o pistas lexicas negativas
#   positivo  -> estrellas 4-5 y pistas lexicas positivas
#   neutro    -> estrellas 3 y pocas pistas (lo mas dificil de minar)
#
# Salidas:
#   data/candidatos_aspectos_para_anotacion.csv  (cola, label vacia)
#   outputs/reports/disponibilidad_aspectos.csv  (deficit vs disponible por celda)
# ============================================================
from pathlib import Path
import re, unicodedata
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA, REP = BASE / "data", BASE / "outputs" / "reports"
REP.mkdir(parents=True, exist_ok=True)

ASPECTOS_FOCO = ["clima", "aforo_multitudes", "limpieza"]
TARGET_POR_POLARIDAD = 100      # objetivo de ejemplos por (aspecto x polaridad)
OVERSAMPLE = 2.0                # se minan ~2x el deficit (no todos se confirmaran)
POLARIDADES = ["negativo", "neutro", "positivo"]

# Lexicos de opinion ESPECIFICOS por aspecto: la pista debe ser SOBRE el aspecto,
# no sobre otro aspecto de la misma resena (reduce el ruido de las pistas globales).
CUE_ASP = {
    "clima": {
        "neg": ["frio","friolento","calor","caloroso","sofocante","lluvia","lluvioso","llovio","nublado",
                "neblina","humedad","humedo","viento","ventoso","mal clima","cold","hot","humid","rain",
                "rainy","cloudy","foggy","windy","bad weather"],
        "pos": ["buen clima","soleado","sol","despejado","fresco","calido","agradable clima","clima ideal",
                "clima perfecto","buen tiempo","sunny","warm","nice weather","clear sky","pleasant weather","mild"],
    },
    "limpieza": {
        "neg": ["sucio","sucia","mugre","mugriento","basura","desaseo","desaseado","descuidado","abandonado",
                "huele mal","hediondo","apesta","banos sucios","dirty","filthy","trash","garbage","smelly",
                "messy","unclean","stinks","poorly maintained"],
        "pos": ["limpio","limpia","impecable","ordenado","aseado","bien cuidado","reluciente","pulcro",
                "clean","spotless","tidy","well kept","well maintained","neat","immaculate"],
    },
    "aforo_multitudes": {
        "neg": ["lleno","llenisimo","saturado","aglomeracion","aglomerado","gentio","tumulto","multitud",
                "colas","cola larga","mucha gente","atestado","apretado","crowded","overcrowded","packed",
                "queue","long line","too many people","jam packed","mobbed"],
        "pos": ["tranquilo","poca gente","sin colas","sin gente","vacio","espacioso","despejado","sin aglomeracion",
                "uncrowded","quiet","not crowded","no queue","no lines","peaceful","spacious","empty","few people"],
    },
}

def norm(t):
    t = str(t).lower(); t = unicodedata.normalize("NFKD", t)
    return "".join(c for c in t if not unicodedata.combining(c))
CUE_ASP_N = {a: {p: [norm(c) for c in v] for p, v in d.items()} for a, d in CUE_ASP.items()}
def cues(texto, lista):
    t = norm(texto); h = [c for c in lista if c in t]; return len(h), ";".join(sorted(set(h))[:5])

# --- Carga ---
absa = pd.read_csv(BASE / "outputs/predictions/tourism_reviews_clean_absa_ready.csv", encoding="utf-8-sig")
clean = pd.read_csv(DATA / "tourism_reviews_clean.csv", encoding="utf-8-sig", usecols=["review_uid", "stars", "sentiment_by_stars"])
gold = pd.read_csv(DATA / "gold_set_v3.csv", encoding="utf-8-sig"); gold["label"] = gold["label"].str.lower()
gold_pairs = set(zip(gold["review_uid"], gold["aspecto"]))

absa = absa.merge(clean, on="review_uid", how="left")
absa["stars"] = pd.to_numeric(absa["stars"], errors="coerce")

def score_polaridad(df, pol, aspecto):
    d = df.copy()
    lex = CUE_ASP_N[aspecto]
    nneg = d["text_clean"].apply(lambda x: cues(x, lex["neg"]))
    npos = d["text_clean"].apply(lambda x: cues(x, lex["pos"]))
    d["n_neg"], d["cue_neg"] = [a[0] for a in nneg], [a[1] for a in nneg]
    d["n_pos"], d["cue_pos"] = [a[0] for a in npos], [a[1] for a in npos]
    if pol == "negativo":
        # exige pista NEGATIVA del aspecto (no solo estrella baja, que puede ser por otro aspecto)
        d = d[d["n_neg"] >= 1]
        d["score"] = d["n_neg"] + d["stars"].map({1: 1.5, 2: 1}).fillna(0)
        d["calidad"] = ((d["n_neg"] >= 2) | (d["stars"].isin([1, 2]))).map({True: "alta", False: "media"})
    elif pol == "positivo":
        # exige pista POSITIVA del aspecto y sin pista negativa del aspecto
        d = d[(d["n_pos"] >= 1) & (d["n_neg"] == 0)]
        d["score"] = d["n_pos"] + d["stars"].map({5: 1, 4: 0.5}).fillna(0)
        d["calidad"] = (d["n_pos"] >= 2).map({True: "alta", False: "media"})
    else:  # neutro: menciona el aspecto SIN pista de opinion (ni pos ni neg del aspecto)
        d = d[(d["n_neg"] == 0) & (d["n_pos"] == 0)]
        d["score"] = (d["stars"] == 3).astype(int) * 2 + 1
        d["calidad"] = (d["stars"] == 3).map({True: "alta", False: "media"})
    return d.sort_values("score", ascending=False)

filas, disp = [], []
for asp in ASPECTOS_FOCO:
    en_gold = gold[gold.aspecto == asp]["label"].value_counts()
    pool_asp = absa[absa.aspecto == asp].copy()
    pool_asp = pool_asp[~pool_asp.apply(lambda r: (r["review_uid"], asp) in gold_pairs, axis=1)]
    pool_asp = pool_asp.drop_duplicates(subset=["review_uid"])
    for pol in POLARIDADES:
        actual = int(en_gold.get(pol, 0))
        deficit = max(0, TARGET_POR_POLARIDAD - actual)
        objetivo = int(round(deficit * OVERSAMPLE))
        cand = score_polaridad(pool_asp, pol, asp)
        disponibles = len(cand)
        sel = []
        if objetivo > 0 and disponibles > 0:
            grupos = {d: g.to_dict("records") for d, g in cand.groupby("destination")}
            punt = {d: 0 for d in grupos}
            while len(sel) < objetivo and any(punt[d] < len(grupos[d]) for d in grupos):
                for d in list(grupos):
                    if punt[d] < len(grupos[d]):
                        sel.append(grupos[d][punt[d]]); punt[d] += 1
                        if len(sel) >= objetivo: break
        for r in sel:
            filas.append({"review_uid": r["review_uid"], "destination": r["destination"],
                          "language_review": r.get("language_review", ""), "aspecto": asp,
                          "polaridad_objetivo": pol, "stars": r["stars"], "calidad_senal": r["calidad"],
                          "text_clean": r["text_clean"], "input_modelo": r.get("input_modelo", ""), "label": ""})
        disp.append({"aspecto": asp, "polaridad": pol, "en_gold": actual, "target": TARGET_POR_POLARIDAD,
                     "deficit": deficit, "a_minar": objetivo, "disponibles": disponibles,
                     "incluidos": len(sel), "suficiente": disponibles >= deficit})

cands = pd.DataFrame(filas).drop_duplicates(subset=["review_uid", "aspecto"])
cands.to_csv(DATA / "candidatos_aspectos_para_anotacion.csv", index=False, encoding="utf-8-sig")
dispdf = pd.DataFrame(disp); dispdf.to_csv(REP / "disponibilidad_aspectos.csv", index=False, encoding="utf-8-sig")

print("=== DISPONIBILIDAD por aspecto x polaridad ===")
print(dispdf.to_string(index=False))
print(f"\nTotal candidatos en la cola: {len(cands)}")
print("Por aspecto x polaridad objetivo:")
print(cands.groupby(["aspecto", "polaridad_objetivo"]).size().to_string())
print("\nArchivo:", DATA / "candidatos_aspectos_para_anotacion.csv")
