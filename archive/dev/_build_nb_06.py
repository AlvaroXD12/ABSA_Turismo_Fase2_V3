# -*- coding: utf-8 -*-
# Construye 03_reporte_absa_xlmr_bert_gold_v4.ipynb: NOTEBOOK-REPORTE.
# No entrena (eso lo hacen los scripts entrenar_modelo_v4.py). Carga artefactos,
# compara XLM-R (principal) vs BERT (base) + TF-IDF, da veredicto, genera la matriz
# con el modelo final y grafica todo.
import json
from pathlib import Path
cells = []
def md(t):  cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})

md(r"""
# 03 — Reporte ABSA gold v4: XLM-R (principal) vs BERT + TextCNN (base oficial)

**Notebook-reporte.** El entrenamiento (lo pesado) se hace en scripts separados, **uno por
modelo y en su propio proceso** (GPU fresca, sin OOM por acumulación). Este notebook **carga
los artefactos** y produce el reporte: comparación, métricas, veredicto y la matriz ABSA.

### Cómo ejecutar (en orden)
```bash
python scripts/entrenar_modelo_v4.py --model xlmr   # candidato principal (hace la búsqueda de HP)
python scripts/entrenar_modelo_v4.py --model bert   # base oficial (reutiliza el HP)
```
Luego abre este notebook → **Restart & Run All** (carga los resultados y grafica).
- **Spec:** [`specs/modulo-absa-fase2.md`](../specs/modulo-absa-fase2.md) · Umbral: F1-macro ≥ 0.70 + mínimos por clase.
""")

code(r"""
import sys
from pathlib import Path
BASE = Path.cwd().parent if Path.cwd().name.lower() == "notebooks" else Path.cwd()
sys.path.append(str(BASE / "scripts"))
import absa_common as ac
from generar_matriz_absa import build_matrix
import numpy as np, pandas as pd, json
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
pd.set_option("display.max_columns", 80); pd.set_option("display.width", 220)
LABELS = ac.LABELS; COLOR = {"negativo":"#d62728","neutro":"#7f7f7f","positivo":"#2ca02c"}
MAX_CORPUS_INFER = None   # None = corpus completo; un entero limita (para pruebas)

faltan = [str(ac.art(t,k).relative_to(BASE)) for t in ("xlmr","bert") for k in ("resumen","det","aspecto","preds") if not ac.art(t,k).exists()]
if faltan:
    print("⚠️  Faltan artefactos de entrenamiento. Corre primero, en orden:")
    print("    python scripts/entrenar_modelo_v4.py --model xlmr")
    print("    python scripts/entrenar_modelo_v4.py --model bert")
    print("Faltan:", faltan[:6], "..." if len(faltan) > 6 else "")
else:
    print("✅ Artefactos completos. DEVICE:", ac.DEVICE)
""")

md(r"""## 1. Datos y splits (gold v4)""")
code(r"""
train, val, test = ac.load_splits()
gold = pd.read_csv(ac.DATA / f"gold_set_{ac.VER}.csv", encoding="utf-8-sig")
print(f"gold {ac.VER}: {len(gold)} | train {len(train)} | val {len(val)} | test {len(test)}")
""")

md(r"""## 2. Verificación de no fuga por `review_uid`""")
code(r"""
s = {n: set(d.review_uid) for n, d in [("tr",train),("va",val),("te",test)]}
fuga = {"tr∩va": len(s["tr"]&s["va"]), "tr∩te": len(s["tr"]&s["te"]), "va∩te": len(s["va"]&s["te"])}
print(fuga); assert sum(fuga.values()) == 0; print("✅ Sin fuga.")
""")

md(r"""## 3. Distribución por split (polaridad y aspecto)""")
code(r"""
alld = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
def tabla(col): return alld.groupby([col,"split"]).size().unstack("split", fill_value=0).reindex(columns=["train","val","test"], fill_value=0)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
(tabla("label")/tabla("label").sum()*100).reindex(LABELS).T.plot(kind="bar", ax=ax[0], color=[COLOR[l] for l in LABELS])
ax[0].set_title("Polaridad por split (%)"); ax[0].tick_params(axis="x", rotation=0)
tabla("aspecto").plot(kind="barh", ax=ax[1]); ax[1].set_title("Aspectos por split")
plt.tight_layout(); plt.show()
""")

md(r"""## 4. Resultados por semilla y HP elegidos por validación""")
code(r"""
print("XLM-R (principal):"); display(pd.read_csv(ac.art("xlmr","det")))
print("BERT (base PPI):"); display(pd.read_csv(ac.art("bert","det")))
if ac.HP_FILE.exists(): print("HP por validación:"); display(pd.read_csv(ac.HP_FILE))
""")

md(r"""## 5. Curvas de entrenamiento/validación (XLM-R, por época)""")
code(r"""
h = pd.read_csv(ac.art("xlmr","hist")); fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
ax[0].plot(h.epoch, h.train_loss, "o-", label="train"); ax[0].plot(h.epoch, h.val_loss, "s-", label="val")
ax[0].set_title("Loss"); ax[0].set_xlabel("época"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(h.epoch, h.train_f1_macro, "o-", label="train"); ax[1].plot(h.epoch, h.val_f1_macro, "s-", label="val")
ax[1].set_title("F1-macro"); ax[1].set_xlabel("época"); ax[1].set_ylim(0,1); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
""")

md(r"""## 6. Media ± desviación estándar (estabilidad)""")
code(r"""
res = pd.concat([pd.read_csv(ac.art("xlmr","resumen")), pd.read_csv(ac.art("bert","resumen"))], ignore_index=True)
display(res)
for _, r in res.iterrows():
    print(f"  {r['modelo']}: F1-macro {r['media_f1_macro']:.4f} ± {r['std_f1_macro']:.4f} -> {'ESTABLE' if r['estable_std<=0.03'] else 'INESTABLE'}")
""")

md(r"""## 7. Comparación de modelos (selección mínima)

Baseline clásico (TF-IDF + Logistic Regression), base oficial (BERT+TextCNN) y candidato (XLM-R+TextCNN).
""")
code(r"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=20000), LogisticRegression(max_iter=2000, class_weight="balanced"))
pipe.fit(train["input_modelo"], train["label"]); base_f1 = ac.metrics(test["label"].tolist(), list(pipe.predict(test["input_modelo"])))["f1_macro"]
comp = pd.DataFrame([{"modelo":"TF-IDF + LogReg (baseline)","f1_macro":round(base_f1,4)},
                     {"modelo":"BERT-mult + TextCNN (base PPI)","f1_macro":float(res.set_index('modelo').loc['bert','ensemble_f1_macro'])},
                     {"modelo":"XLM-R + TextCNN (principal)","f1_macro":float(res.set_index('modelo').loc['xlmr','ensemble_f1_macro'])}]).sort_values("f1_macro")
comp.to_csv(ac.REP / f"comparacion_modelos_{ac.VER}.csv", index=False, encoding="utf-8-sig")
plt.figure(figsize=(9,4)); plt.barh(comp.modelo, comp.f1_macro, color="#4c72b0"); plt.xlim(0,1)
plt.axvline(0.70, ls="--", color="green", label="objetivo 0.70"); plt.title("F1-macro (test)")
for i,(_,r) in enumerate(comp.iterrows()): plt.text(r.f1_macro+.01, i, f"{r.f1_macro:.3f}", va="center")
plt.legend(); plt.tight_layout(); plt.show(); display(comp)
""")

md(r"""## 8. Métricas obligatorias del candidato principal (XLM-R)""")
code(r"""
pt = pd.read_csv(ac.art("xlmr","preds")); yt, yp = pt.y_true.tolist(), pt.y_pred.tolist()
clsrep = pd.DataFrame(classification_report(yt, yp, labels=LABELS, output_dict=True, zero_division=0)).T
clsrep.to_csv(ac.REP / f"classification_report_{ac.VER}.csv", encoding="utf-8-sig")
pa = pd.read_csv(ac.art("xlmr","aspecto"))
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
cm = confusion_matrix(yt, yp, labels=LABELS); ax[0,0].imshow(cm, cmap="Blues")
ax[0,0].set_xticks(range(3)); ax[0,0].set_xticklabels(LABELS, rotation=45); ax[0,0].set_yticks(range(3)); ax[0,0].set_yticklabels(LABELS)
ax[0,0].set_title("Matriz de confusión (XLM-R)"); ax[0,0].set_ylabel("real"); ax[0,0].set_xlabel("pred")
for i in range(3):
    for j in range(3): ax[0,0].text(j, i, cm[i,j], ha="center", va="center")
f1c=[clsrep.loc[l,"f1-score"] for l in LABELS]; ax[0,1].bar(LABELS, f1c, color=[COLOR[l] for l in LABELS]); ax[0,1].axhline(0.6, ls="--", color="gray"); ax[0,1].set_ylim(0,1); ax[0,1].set_title("F1 por clase")
for i,v in enumerate(f1c): ax[0,1].text(i, v+.02, f"{v:.2f}", ha="center")
rec=[clsrep.loc[l,"recall"] for l in LABELS]; ax[1,0].bar(LABELS, rec, color=["#d62728","#ccc","#ccc"]); ax[1,0].axhline(0.6, ls="--", color="gray"); ax[1,0].set_ylim(0,1); ax[1,0].set_title("Recall por clase (negativo resaltado)")
for i,v in enumerate(rec): ax[1,0].text(i, v+.02, f"{v:.2f}", ha="center")
pas=pa.sort_values("f1_macro"); ax[1,1].barh(pas.aspecto, pas.f1_macro, color="#1f77b4"); ax[1,1].set_xlim(0,1); ax[1,1].set_title("F1 por aspecto (XLM-R)")
plt.tight_layout(); plt.show(); display(clsrep.round(3))
""")

md(r"""## 9. Comparación v3 vs v4 por aspecto (referencia de evolución)

**Nota:** v3 y v4 tienen tests distintos (re-partición); es referencia de evolución, no prueba estricta.
""")
code(r"""
pa4 = pd.read_csv(ac.art("xlmr","aspecto")).rename(columns={"f1_macro":"v4_xlmr"})[["aspecto","v4_xlmr"]]
v3p = ac.REP / "por_aspecto_v3.csv"
if v3p.exists():
    pa3 = pd.read_csv(v3p).rename(columns={"f1_macro":"v3_bert"})[["aspecto","v3_bert"]]
    cmp = pa3.merge(pa4, on="aspecto", how="outer").fillna(0).sort_values("v4_xlmr"); foco=["clima","aforo_multitudes","limpieza"]
    y=np.arange(len(cmp)); plt.figure(figsize=(10,5))
    plt.barh(y-0.2, cmp.v3_bert, 0.4, label="v3 (BERT)", color="#bbb")
    plt.barh(y+0.2, cmp.v4_xlmr, 0.4, label="v4 (XLM-R)", color=["#ff7f0e" if a in foco else "#1f77b4" for a in cmp.aspecto])
    plt.yticks(y, [a+(" ★" if a in foco else "") for a in cmp.aspecto]); plt.xlim(0,1); plt.legend(); plt.title("F1 por aspecto: v3 vs v4 (★ reforzados)")
    plt.tight_layout(); plt.show(); display(cmp.round(3))
""")

md(r"""## 10. Veredicto y selección del modelo final

XLM-R se adopta como modelo final solo si cumple la spec de forma **estable** (≥0.70, mínimos por
clase, std ≤ 0.03). Si no, se mantiene la mejor versión defendible.
""")
code(r"""
rr = res.set_index("modelo")
def veredicto(tag):
    r = rr.loc[tag]; chk = {"F1-macro≥0.70": r["ensemble_f1_macro"]>=ac.TH_MACRO, "neg F1≥0.60": r["ensemble_f1_negativo"]>=ac.TH_NEG_F1,
        "neg recall≥0.60": r["ensemble_recall_negativo"]>=ac.TH_NEG_REC, "neu F1≥0.60": r["ensemble_f1_neutro"]>=ac.TH_NEU_F1, "estable": bool(r["estable_std<=0.03"])}
    return all(chk.values()), chk, r
for tag in ["xlmr","bert"]:
    ok, chk, r = veredicto(tag)
    print(f"=== {tag.upper()} === F1-macro={r['ensemble_f1_macro']:.3f} neg_F1={r['ensemble_f1_negativo']:.3f} neg_rec={r['ensemble_recall_negativo']:.3f} neu_F1={r['ensemble_f1_neutro']:.3f} std={r['std_f1_macro']:.4f}")
    for k,v in chk.items(): print(f"    [{'OK' if v else 'X'}] {k}")
xlmr_ok, _, _ = veredicto("xlmr")
MODELO_FINAL = "xlmr" if xlmr_ok else ("xlmr" if rr.loc["xlmr","ensemble_f1_macro"]>=rr.loc["bert","ensemble_f1_macro"] else "bert")
VEREDICTO = "ÉXITO TÉCNICO (XLM-R cumple la spec de forma estable)" if xlmr_ok else "VERSIÓN BASE DEFENDIBLE (XLM-R como mejora exploratoria)"
print(f"\nMODELO PARA LA MATRIZ: {MODELO_FINAL.upper()} | {VEREDICTO}")
""")

md(r"""## 11. Matriz destino-aspecto-sentimiento (modelo final)

Inferencia del ensemble del modelo seleccionado sobre el corpus completo.
""")
code(r"""
matr_csv = ac.MATR_DIR / "matriz_destino_aspecto_sentimiento.csv"
corpus = pd.read_csv(BASE / "outputs/predictions/tourism_reviews_clean_absa_ready.csv", encoding="utf-8-sig")
if MAX_CORPUS_INFER: corpus = corpus.head(MAX_CORPUS_INFER).copy()
if "input_modelo" not in corpus.columns or corpus["input_modelo"].isna().any():
    corpus["input_modelo"] = "aspecto: " + corpus["aspecto"].astype(str) + " reseña: " + corpus["text_clean"].astype(str)
mname = ac.MODELOS[MODELO_FINAL]; bias = np.load(ac.art(MODELO_FINAL,"bias")) if ac.art(MODELO_FINAL,"bias").exists() else np.zeros(3)
tok = AutoTokenizer.from_pretrained(mname)
cl = DataLoader(ac.ABSADataset(corpus["input_modelo"], None, tok), batch_size=max(ac.BATCH,16))
cp = np.zeros((len(corpus),3)); nseed = 0
for sd in ac.SEEDS:
    mp = ac.MODELS_DIR / f"modelo_{MODELO_FINAL}_seed{sd}_{ac.VER}.pt"
    if not mp.exists(): continue
    model = ac.TextCNN(mname).to(ac.DEVICE); model.load_state_dict(torch.load(mp, map_location="cpu")); model.eval(); nseed+=1; pos=0
    with torch.no_grad():
        for b in cl:
            with torch.autocast("cuda", enabled=ac.USE_AMP):
                lo = model(b["input_ids"].to(ac.DEVICE), b["attention_mask"].to(ac.DEVICE))
            pr = torch.softmax(lo.float(),1).cpu().numpy(); cp[pos:pos+len(pr)] += pr; pos+=len(pr)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
cp /= max(nseed,1); corpus["label_pred"] = ac.apply_bias(cp, bias)
corpus[["review_uid","destination","aspecto","label_pred"]].to_csv(ac.PRED_DIR / f"predicciones_corpus_{ac.VER}.csv", index=False, encoding="utf-8-sig")
matriz = build_matrix(corpus[["review_uid","destination","aspecto","label_pred"]])
matriz.to_csv(matr_csv, index=False, encoding="utf-8-sig"); matriz.to_json(ac.MATR_DIR / "matriz_destino_aspecto_sentimiento.json", orient="records", force_ascii=False, indent=2)
print("Matriz:", matriz.shape, "| modelo:", MODELO_FINAL.upper(), "| semillas usadas:", nseed); display(matriz.head(12))
""")

md(r"""## 12. Vistas de la matriz: niveles de evidencia y heatmap destino × aspecto""")
code(r"""
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios":[1,2.2]})
orden=["sin datos","insuficiente","baja","suficiente"]; ev=matriz["evidence_status"].value_counts().reindex(orden, fill_value=0)
ax[0].bar(ev.index, ev.values, color=["#ccc","#f4a582","#fdb863","#4daf4a"]); ax[0].set_title("Niveles de evidencia"); ax[0].tick_params(axis="x", rotation=20)
for i,v in enumerate(ev.values): ax[0].text(i, v+0.5, int(v), ha="center")
piv = matriz.pivot(index="destination", columns="aspecto", values="score_ajustado")
im = ax[1].imshow(piv.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
ax[1].set_xticks(range(len(piv.columns))); ax[1].set_xticklabels(piv.columns, rotation=45, ha="right")
ax[1].set_yticks(range(len(piv.index))); ax[1].set_yticklabels(piv.index, fontsize=8); ax[1].set_title("Heatmap destino × aspecto (score_ajustado)")
fig.colorbar(im, ax=ax[1], shrink=0.8); plt.tight_layout(); plt.show()
""")

md(r"""## 13. Conclusión""")
code(r"""
rx = rr.loc["xlmr"]
print("CONCLUSIÓN"); print("-"*64)
print(f"Gold {ac.VER} ({len(gold)} ítems, split 70/15/15 sin fuga, {len(ac.SEEDS)} semillas, HP por validación).")
print(f"XLM-R (principal): F1-macro={rx['ensemble_f1_macro']:.3f} (media {rx['media_f1_macro']:.3f}±{rx['std_f1_macro']:.4f}), "
      f"neg F1={rx['ensemble_f1_negativo']:.3f} recall={rx['ensemble_recall_negativo']:.3f}, neu F1={rx['ensemble_f1_neutro']:.3f}")
print(f"BERT (base PPI): F1-macro={rr.loc['bert','ensemble_f1_macro']:.3f}")
print(f"VEREDICTO: {VEREDICTO} | Matriz generada con: {MODELO_FINAL.upper()}")
""")

md(r"""
**Cierre.** Entrenamiento aislado por modelo (scripts en procesos separados → sin OOM por
acumulación); este notebook es el reporte único. XLM-R = candidato principal; BERT+TextCNN = base
oficial. HP de la clase negativa elegidos en validación. v3↔v4 no comparables de forma estricta
(tests distintos). La matriz final se genera con el modelo que cumple la spec de forma estable.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
      "language_info": {"name":"python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parent.parent / "notebooks" / "03_reporte_absa_xlmr_bert_gold_v4.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook escrito:", out, "| celdas:", len(cells))
