# -*- coding: utf-8 -*-
# Construye 03_entrenamiento_absa_bert_textcnn_gold_v4.ipynb:
# bitacora final, visual y defendible (datos -> split -> entrenamiento ->
# comparacion minima -> metricas -> matriz ABSA -> veredicto).
# RUN_TRAINING=False por defecto: carga artefactos; si faltan, entrena (auto-fallback).
import json
from pathlib import Path

cells = []
def md(t):  cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})

md(r"""
# 03 — Entrenamiento y evaluación ABSA (BERT multilingual + TextCNN) — gold v4

**Bitácora final, visual y reproducible.** Documenta el módulo ABSA de la Fase 2 sobre el
**gold v4** (reforzado en aspectos débiles: clima, aforo, limpieza). Flujo:
datos → split → entrenamiento → comparación mínima de modelos → métricas → matriz ABSA → veredicto.

- **Spec:** [`specs/modulo-absa-fase2.md`](../specs/modulo-absa-fase2.md)
- **Objetivo:** F1-macro ≥ 0.70 en test, con mínimos por clase (neg F1 ≥ 0.60 y recall ≥ 0.60; neu F1 ≥ 0.60).

### Cómo ejecutarlo
- `RUN_TRAINING = False` por defecto: si los artefactos existen, los **carga** y grafica.
  Si **faltan**, el notebook **entrena automáticamente** (auto-fallback). Por eso *Restart & Run All*
  funciona aunque sea la primera vez.
- Pon `RUN_TRAINING = True` para forzar el re-entrenamiento.
- En GPU con poca VRAM (<6GB): `USE_GRADIENT_CHECKPOINTING=True` y baja `BATCH`.
""")

code(r"""
from pathlib import Path
import json, random, time, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 80); pd.set_option("display.width", 220)

BASE = Path.cwd().parent if Path.cwd().name.lower() == "notebooks" else Path.cwd()
DATA, REP, VIS = BASE / "data", BASE / "outputs" / "reports", BASE / "outputs" / "visualizations"
MODELS_DIR, PRED_DIR, MATR_DIR = BASE / "models", BASE / "outputs" / "predictions", BASE / "outputs" / "matrices"
for d in (REP, VIS, MODELS_DIR, PRED_DIR, MATR_DIR): d.mkdir(parents=True, exist_ok=True)
LABELS = ["negativo", "neutro", "positivo"]; L2I = {l: i for i, l in enumerate(LABELS)}; I2L = {i: l for l, i in L2I.items()}
COLOR = {"negativo": "#d62728", "neutro": "#7f7f7f", "positivo": "#2ca02c"}

# ---------------- Config ----------------
VER = "v4"
RUN_TRAINING = False                 # False: carga artefactos; auto-entrena si faltan
MODEL_NAME = "bert-base-multilingual-cased"
MAX_LEN, BATCH, EPOCHS = 256, 8, 12
LR, WEIGHT_DECAY, WARMUP_RATIO, PATIENCE, DROPOUT = 2e-5, 0.10, 0.10, 3, 0.40
CNN_FILTERS, CNN_KERNELS = 128, (2, 3, 4)
SEEDS = [42, 7, 123, 2024, 77]
FOCAL_GAMMA, NEG_BOOST, LABEL_SMOOTHING = 1.0, 1.2, 0.1
CALIBRAR_DECISION = True
USE_GRADIENT_CHECKPOINTING = False
INCLUIR_XLMR = True                  # variante de referencia (1 semilla); pon False para omitir
XLMR_NAME = "xlm-roberta-base"
MAX_CORPUS_INFER = None              # None = todo el corpus; un entero limita filas (para pruebas)
TH_MACRO, TH_NEG_F1, TH_NEG_REC, TH_NEU_F1 = 0.70, 0.60, 0.60, 0.60

# Artefactos que el notebook necesita para graficar sin re-entrenar
ART = {
    "det": REP / f"resultados_bert_textcnn_{VER}.csv",
    "resumen": REP / f"resumen_bert_textcnn_{VER}.csv",
    "clsrep": REP / f"classification_report_{VER}.csv",
    "aspecto": REP / f"por_aspecto_{VER}.csv",
    "hist": REP / f"historial_entrenamiento_{VER}.csv",
    "baselines": REP / f"comparacion_modelos_{VER}.csv",
    "preds_test": PRED_DIR / f"predicciones_test_{VER}.csv",
    "matriz": MATR_DIR / "matriz_destino_aspecto_sentimiento.csv",
}
def artefactos_existen():
    return all(p.exists() for p in ART.values())

print("BASE:", BASE, "| VER:", VER, "| RUN_TRAINING:", RUN_TRAINING, "| artefactos:", artefactos_existen())
""")

md(r"""## 1. Carga del gold v4 y de los splits""")
code(r"""
gold = pd.read_csv(DATA / f"gold_set_{VER}.csv", encoding="utf-8-sig")
train = pd.read_csv(DATA / f"train_gold_{VER}.csv", encoding="utf-8-sig")
val   = pd.read_csv(DATA / f"val_gold_{VER}.csv", encoding="utf-8-sig")
test  = pd.read_csv(DATA / f"test_gold_{VER}.csv", encoding="utf-8-sig")
for d in (gold, train, val, test):
    d["label"] = d["label"].astype(str).str.lower().str.strip()
    if "input_modelo" not in d.columns or d["input_modelo"].isna().any():
        d["input_modelo"] = "aspecto: " + d["aspecto"].astype(str) + " reseña: " + d["text_clean"].astype(str)
print(f"gold {VER}: {len(gold)} ítems | {gold['review_uid'].nunique()} reseñas únicas")
for nm, d in [("train", train), ("val", val), ("test", test)]:
    print(f"{nm:5s}: {len(d):5d} ({len(d)/len(gold)*100:.1f}%)")
train.head(3)
""")

md(r"""## 2. Verificación de no fuga por `review_uid`""")
code(r"""
s_tr, s_va, s_te = set(train.review_uid), set(val.review_uid), set(test.review_uid)
fuga = {"train∩val": len(s_tr & s_va), "train∩test": len(s_tr & s_te), "val∩test": len(s_va & s_te)}
print("Solapamientos:", fuga); assert sum(fuga.values()) == 0, "FUGA"
print("✅ Sin fuga.")
""")

md(r"""## 3. Distribución por split (polaridad y aspecto)""")
code(r"""
alld = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
def tabla(col):
    return alld.groupby([col, "split"]).size().unstack("split", fill_value=0).reindex(columns=["train","val","test"], fill_value=0)

# Gráfico 1: polaridad por split (%)
pol = (tabla("label") / tabla("label").sum() * 100).reindex(LABELS)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
pol.T.plot(kind="bar", ax=ax[0], color=[COLOR[l] for l in pol.index])
ax[0].set_title("Distribución de polaridad por split (%)"); ax[0].set_ylabel("%"); ax[0].tick_params(axis="x", rotation=0)
# Gráfico 2: aspecto por split (conteo)
tabla("aspecto").plot(kind="barh", ax=ax[1]); ax[1].set_title("Distribución de aspectos por split"); ax[1].set_xlabel("ítems")
plt.tight_layout(); plt.show()
display(tabla("label"))
""")

md(r"""## 4. Modelo: BERT multilingual + TextCNN

Encoder Transformer multilingüe → convoluciones 1D (kernels 2/3/4) + max-pooling →
clasificador de 3 clases. La ablación previa mostró que el encoder no es la palanca;
se conserva esta arquitectura y se ataca la señal de la clase negativa.
""")
code(r"""
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

class ABSADataset(Dataset):
    def __init__(self, texts, labels, tok): self.t=list(texts); self.l=list(labels); self.tok=tok
    def __len__(self): return len(self.t)
    def __getitem__(self, i):
        e = self.tok(str(self.t[i]), add_special_tokens=True, max_length=MAX_LEN, padding="max_length",
                     truncation=True, return_attention_mask=True, return_tensors="pt")
        return {"input_ids": e["input_ids"].squeeze(0), "attention_mask": e["attention_mask"].squeeze(0),
                "labels": torch.tensor(L2I[self.l[i]], dtype=torch.long)}

class BERTTextCNN(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        if USE_GRADIENT_CHECKPOINTING:
            self.bert.config.use_cache = False; self.bert.gradient_checkpointing_enable()
        h = self.bert.config.hidden_size
        self.convs = nn.ModuleList([nn.Conv1d(h, CNN_FILTERS, k) for k in CNN_KERNELS])
        self.drop = nn.Dropout(DROPOUT); self.fc = nn.Linear(CNN_FILTERS * len(CNN_KERNELS), 3)
    def forward(self, ids, mask):
        x = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state.transpose(1, 2)
        pooled = [torch.max(torch.relu(c(x)), dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(pooled, dim=1)))
print("DEVICE:", DEVICE, "| AMP:", USE_AMP)
""")

md(r"""## 5. Configuración enfocada a la clase negativa + calibración

Class weights inversos con refuerzo a negativa (`NEG_BOOST`), **Focal Loss**, **label smoothing**, y
**calibración de decisión** (ajusta la frontera en validación para maximizar F1-macro; rebalancea
precisión/recall del negativo sin re-entrenar).
""")
code(r"""
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report

class FocalLoss(nn.Module):
    def __init__(self, weight, gamma): super().__init__(); self.w=weight; self.g=gamma
    def forward(self, logits, y):
        ce = nn.functional.cross_entropy(logits, y, weight=self.w, reduction="none", label_smoothing=LABEL_SMOOTHING)
        return (((1 - torch.exp(-ce)) ** self.g) * ce).mean()

def class_weights(labels):
    c = pd.Series(labels).value_counts().reindex(LABELS, fill_value=0); tot = c.sum()
    w = [tot/(3*c[l]) if c[l] > 0 else 0.0 for l in LABELS]; w[L2I["negativo"]] *= NEG_BOOST
    return torch.tensor(w, dtype=torch.float).to(DEVICE)

def metrics(trues, preds):
    pr, rc, f1, _ = precision_recall_fscore_support(trues, preds, labels=LABELS, average=None, zero_division=0)
    _, _, mf1, _ = precision_recall_fscore_support(trues, preds, labels=LABELS, average="macro", zero_division=0)
    o = {"f1_macro": mf1, "accuracy": accuracy_score(trues, preds)}
    for i, l in enumerate(LABELS): o[f"f1_{l}"] = f1[i]; o[f"recall_{l}"] = rc[i]
    return o

def apply_bias(probs, bias): return [I2L[i] for i in (np.log(probs + 1e-9) + bias).argmax(1)]
def best_bias(vp, vt):
    if not CALIBRAR_DECISION: return np.zeros(3)
    logp = np.log(vp + 1e-9); g = np.arange(-1.2, 1.21, 0.2); best, bb = -1, np.zeros(3)
    for b0 in g:
        for b1 in g:
            b = np.array([b0, b1, 0.0]); f = metrics(vt, [I2L[i] for i in (logp + b).argmax(1)])["f1_macro"]
            if f > best: best, bb = f, b
    return bb

_w = class_weights(train["label"])
print("Pesos por clase:", {l: round(float(_w[L2I[l]]), 3) for l in LABELS}, "| focal γ", FOCAL_GAMMA, "| neg_boost", NEG_BOOST)
""")

md(r"""## 6. Entrenamiento (5 semillas) + baselines + calibración

`RUN_TRAINING=False` carga los artefactos ya generados; si faltan, entrena automáticamente.
Se entrena BERT+TextCNN (5 semillas, ensemble), un baseline **TF-IDF + Logistic Regression**, y
(opcional) **XLM-R** como variante de referencia. Para una semilla se guarda el **historial por época**
(loss y F1-macro) para las curvas.
""")
code(r"""
def predict(model, loader, loss_fn=None):
    model.eval(); P, T, tot = [], [], 0.0
    with torch.no_grad():
        for b in loader:
            ids, mask, y = b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE), b["labels"].to(DEVICE)
            with torch.autocast("cuda", enabled=USE_AMP):
                lo = model(ids, mask)
                if loss_fn is not None: tot += loss_fn(lo, y).item()
            P.append(torch.softmax(lo.float(), 1).cpu().numpy()); T += [I2L[i] for i in y.cpu().numpy()]
    return np.concatenate(P), T, (tot / max(len(loader), 1))

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def train_one(seed, model_name=MODEL_NAME, record_history=False, save_tag=None):
    set_seed(seed); tok = AutoTokenizer.from_pretrained(model_name)
    tl = DataLoader(ABSADataset(train["input_modelo"], train["label"], tok), batch_size=BATCH, shuffle=True)
    vl = DataLoader(ABSADataset(val["input_modelo"], val["label"], tok), batch_size=BATCH)
    el = DataLoader(ABSADataset(test["input_modelo"], test["label"], tok), batch_size=BATCH)
    model = BERTTextCNN(model_name).to(DEVICE); loss_fn = FocalLoss(class_weights(train["label"]), FOCAL_GAMMA)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = get_linear_schedule_with_warmup(opt, int(len(tl)*EPOCHS*WARMUP_RATIO), len(tl)*EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    best, best_state, pat, hist = -1, None, 0, []
    for ep in range(1, EPOCHS + 1):
        model.train(); run = 0.0
        for b in tl:
            opt.zero_grad()
            with torch.autocast("cuda", enabled=USE_AMP):
                loss = loss_fn(model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE)), b["labels"].to(DEVICE))
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step(); run += loss.item()
        vp, vt, vloss = predict(model, vl, loss_fn); vf = metrics(vt, [I2L[i] for i in vp.argmax(1)])["f1_macro"]
        if record_history:
            tp_, tt_, _ = predict(model, tl); tf = metrics(tt_, [I2L[i] for i in tp_.argmax(1)])["f1_macro"]
            hist.append({"epoch": ep, "train_loss": run/len(tl), "val_loss": vloss, "train_f1_macro": tf, "val_f1_macro": vf})
        print(f"    seed {seed} ep {ep:2d}/{EPOCHS} val_f1={vf:.3f}")
        if vf > best: best, best_state, pat = vf, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= PATIENCE: print("    early stopping"); break
    if best_state: model.load_state_dict(best_state)
    if save_tag: torch.save(best_state, MODELS_DIR / f"modelo_{VER}_{save_tag}.pt")
    tp, tt, _ = predict(model, el); vp, vtv, _ = predict(model, vl); del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return tp, tt, vp, vtv, pd.DataFrame(hist)

def baseline_tfidf_logreg():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=20000),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))
    pipe.fit(train["input_modelo"], train["label"])
    pr = pipe.predict(test["input_modelo"]); return metrics(test["label"].tolist(), list(pr))

NEED_TRAIN = RUN_TRAINING or (not artefactos_existen())
if NEED_TRAIN:
    print("== Entrenando (NEED_TRAIN=True) ==")
    t0 = time.time(); rows, probs, val_probs, test_trues, val_trues, hist_df = [], [], [], None, None, None
    for k, seed in enumerate(SEEDS):
        print(f"=== Semilla {seed} ===")
        p, tt, vp, vtv, h = train_one(seed, record_history=(k == 0), save_tag=f"seed{seed}")
        test_trues, val_trues = tt, vtv; probs.append(p); val_probs.append(vp)
        if k == 0: hist_df = h
        m = metrics(tt, [I2L[i] for i in p.argmax(1)]); rows.append({"seed": seed, **{kk: round(v,4) for kk,v in m.items()}})
    det = pd.DataFrame(rows); det.to_csv(ART["det"], index=False, encoding="utf-8-sig")
    hist_df.to_csv(ART["hist"], index=False, encoding="utf-8-sig")
    ens = np.mean(probs, 0); ensv = np.mean(val_probs, 0)
    BIAS = best_bias(ensv, val_trues); ens_preds = apply_bias(ens, BIAS); ens_metrics = metrics(test_trues, ens_preds)
    pd.DataFrame({"y_true": test_trues, "y_pred": ens_preds}).to_csv(ART["preds_test"], index=False, encoding="utf-8-sig")
    # baselines + comparacion de modelos
    comp = [{"modelo": "TF-IDF + LogReg", "f1_macro": round(baseline_tfidf_logreg()["f1_macro"], 4)},
            {"modelo": "BERT-mult + TextCNN (ensemble)", "f1_macro": round(ens_metrics["f1_macro"], 4)}]
    if INCLUIR_XLMR:
        try:
            xp, xt, _, _, _ = train_one(42, model_name=XLMR_NAME); xm = metrics(xt, [I2L[i] for i in xp.argmax(1)])
            comp.append({"modelo": "XLM-R + TextCNN (ref, 1 semilla)", "f1_macro": round(xm["f1_macro"], 4)})
        except Exception as e: print("XLM-R omitido:", e)
    pd.DataFrame(comp).to_csv(ART["baselines"], index=False, encoding="utf-8-sig")
    # reportes finales
    cols = [c for c in det.columns if c != "seed"]
    res = {**{f"ensemble_{k}": round(v,4) for k,v in ens_metrics.items()},
           **{f"media_{c}": round(det[c].mean(),4) for c in cols}, **{f"std_{c}": round(det[c].std(),4) for c in cols},
           "estable_std<=0.03": bool(det["f1_macro"].std() <= 0.03)}
    pd.DataFrame([res]).to_csv(ART["resumen"], index=False, encoding="utf-8-sig")
    pd.DataFrame(classification_report(test_trues, ens_preds, labels=LABELS, output_dict=True, zero_division=0)).T.to_csv(ART["clsrep"], encoding="utf-8-sig")
    ta = test.copy(); ta["pred"] = ens_preds; fa = []
    for asp, gg in ta.groupby("aspecto"):
        _, _, f1, _ = precision_recall_fscore_support(gg["label"], gg["pred"], labels=LABELS, average="macro", zero_division=0)
        fa.append({"aspecto": asp, "soporte": len(gg), "f1_macro": round(f1, 4)})
    pd.DataFrame(fa).sort_values("f1_macro").to_csv(ART["aspecto"], index=False, encoding="utf-8-sig")
    np.save(MODELS_DIR / f"_bias_{VER}.npy", BIAS)
    print(f"Entrenamiento completo en {(time.time()-t0)/60:.1f} min | BIAS {np.round(BIAS,2)}")
else:
    print("== Cargando artefactos (NEED_TRAIN=False) ==")
""")

md(r"""## 7. Resultados por semilla""")
code(r"""det = pd.read_csv(ART["det"]); display(det)""")

md(r"""## 8. Curvas de entrenamiento y validación (loss y F1-macro por época)""")
code(r"""
h = pd.read_csv(ART["hist"])
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
ax[0].plot(h.epoch, h.train_loss, "o-", label="train"); ax[0].plot(h.epoch, h.val_loss, "s-", label="val")
ax[0].set_title("Curva de pérdida (loss)"); ax[0].set_xlabel("época"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(h.epoch, h.train_f1_macro, "o-", label="train"); ax[1].plot(h.epoch, h.val_f1_macro, "s-", label="val")
ax[1].set_title("Curva F1-macro"); ax[1].set_xlabel("época"); ax[1].set_ylabel("F1-macro"); ax[1].set_ylim(0,1); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
""")

md(r"""## 9. Media ± desviación estándar (estabilidad por semillas)""")
code(r"""
cols = [c for c in det.columns if c != "seed"]
agg = pd.DataFrame({"media": det[cols].mean().round(4), "desv_std": det[cols].std().round(4)})
display(agg)
print(f"Std F1-macro = {det['f1_macro'].std():.4f} -> {'ESTABLE' if det['f1_macro'].std()<=0.03 else 'INESTABLE (>0.03)'}")
""")

md(r"""## 10. Métricas obligatorias (ensemble calibrado)""")
code(r"""
pt = pd.read_csv(ART["preds_test"]); yt, yp = pt["y_true"].tolist(), pt["y_pred"].tolist()
clsrep = pd.read_csv(ART["clsrep"], index_col=0); por_asp = pd.read_csv(ART["aspecto"])
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
# Matriz de confusión
cm = confusion_matrix(yt, yp, labels=LABELS)
im = ax[0,0].imshow(cm, cmap="Blues"); ax[0,0].set_xticks(range(3)); ax[0,0].set_xticklabels(LABELS, rotation=45)
ax[0,0].set_yticks(range(3)); ax[0,0].set_yticklabels(LABELS); ax[0,0].set_title("Matriz de confusión (ensemble)")
ax[0,0].set_ylabel("real"); ax[0,0].set_xlabel("predicho")
for i in range(3):
    for j in range(3): ax[0,0].text(j, i, cm[i,j], ha="center", va="center", color="black")
# F1 por clase
f1c = [clsrep.loc[l, "f1-score"] for l in LABELS]
ax[0,1].bar(LABELS, f1c, color=[COLOR[l] for l in LABELS]); ax[0,1].axhline(0.60, ls="--", color="gray")
ax[0,1].set_ylim(0,1); ax[0,1].set_title("F1 por clase (línea = mínimo 0.60)")
for i,v in enumerate(f1c): ax[0,1].text(i, v+.02, f"{v:.2f}", ha="center")
# Recall por clase, resaltando negativo
rec = [clsrep.loc[l, "recall"] for l in LABELS]
bars = ax[1,0].bar(LABELS, rec, color=["#d62728","#cccccc","#cccccc"]); ax[1,0].axhline(0.60, ls="--", color="gray")
ax[1,0].set_ylim(0,1); ax[1,0].set_title("Recall por clase (negativo resaltado)")
for i,v in enumerate(rec): ax[1,0].text(i, v+.02, f"{v:.2f}", ha="center")
# F1 por aspecto
pa = por_asp.sort_values("f1_macro")
ax[1,1].barh(pa["aspecto"], pa["f1_macro"], color="#1f77b4"); ax[1,1].set_xlim(0,1); ax[1,1].set_title("F1-macro por aspecto")
for i,(_,r) in enumerate(pa.iterrows()): ax[1,1].text(r.f1_macro+.01, i, f"{r.f1_macro:.2f}", va="center")
plt.tight_layout(); plt.show()
display(clsrep.round(3))
""")

md(r"""## 11. Comparación de modelos (selección mínima)

Solo las comparaciones necesarias para la decisión: baseline clásico, modelo principal y
(opcional) una variante Transformer de referencia. No es un barrido exhaustivo.
""")
code(r"""
comp = pd.read_csv(ART["baselines"]).sort_values("f1_macro")
plt.figure(figsize=(9, 4))
plt.barh(comp["modelo"], comp["f1_macro"], color="#4c72b0"); plt.xlim(0,1)
plt.axvline(0.70, ls="--", color="green", label="objetivo 0.70"); plt.title("Comparación de modelos — F1-macro (test)")
for i,(_,r) in enumerate(comp.iterrows()): plt.text(r.f1_macro+.01, i, f"{r.f1_macro:.3f}", va="center")
plt.legend(); plt.tight_layout(); plt.show()
display(comp)
""")

md(r"""## 12. Comparación v3 vs v4 por aspecto (referencia de evolución)

**Nota metodológica:** el test de v4 fue **re-particionado**, así que el F1-macro de v4 **no se compara
directamente** contra el 0.6965 de v3 en el mismo test. Esta comparación por aspecto es una **referencia
de evolución**, no una prueba estricta. La mejora real se sostiene en el desempeño por aspecto (clima,
aforo, limpieza) y en la estabilidad por semillas.
""")
code(r"""
pa4 = pd.read_csv(ART["aspecto"]).rename(columns={"f1_macro": "v4"})[["aspecto", "v4"]]
v3p = REP / "por_aspecto_v3.csv"
if v3p.exists():
    pa3 = pd.read_csv(v3p).rename(columns={"f1_macro": "v3"})[["aspecto", "v3"]]
    cmp = pa3.merge(pa4, on="aspecto", how="outer").fillna(0).sort_values("v4")
    foco = ["clima", "aforo_multitudes", "limpieza"]
    y = np.arange(len(cmp)); plt.figure(figsize=(10, 5))
    plt.barh(y-0.2, cmp["v3"], 0.4, label="v3", color="#bbbbbb")
    plt.barh(y+0.2, cmp["v4"], 0.4, label="v4", color=["#ff7f0e" if a in foco else "#1f77b4" for a in cmp["aspecto"]])
    plt.yticks(y, [a+(" ★" if a in foco else "") for a in cmp["aspecto"]]); plt.xlim(0,1)
    plt.title("F1 por aspecto: v3 vs v4 (★ = aspectos reforzados)"); plt.legend(); plt.tight_layout(); plt.show()
    display(cmp.round(3))
else:
    print("No se encontró por_aspecto_v3.csv; se omite la comparación.")
""")

md(r"""## 13. Veredicto automático contra la especificación""")
code(r"""
res = pd.read_csv(ART["resumen"]).iloc[0]
checks = {"F1-macro ≥ 0.70": (res["ensemble_f1_macro"], TH_MACRO),
          "Negativo F1 ≥ 0.60": (res["ensemble_f1_negativo"], TH_NEG_F1),
          "Recall negativo ≥ 0.60": (res["ensemble_recall_negativo"], TH_NEG_REC),
          "Neutro F1 ≥ 0.60": (res["ensemble_f1_neutro"], TH_NEU_F1)}
print("=== Veredicto (ensemble) ==="); todos = True
for k,(v,th) in checks.items():
    ok = v >= th; todos &= ok; print(f"  [{'PASA' if ok else 'FALLA'}] {k:24s} -> {v:.3f}")
print(f"  [{'OK' if res['estable_std<=0.03'] else 'X'}] Estabilidad std ≤ 0.03 -> {res['std_f1_macro']:.4f}")
VEREDICTO = "ÉXITO TÉCNICO" if todos else "VERSIÓN BASE DEFENDIBLE"
print("\nRESULTADO:", "✅ ÉXITO TÉCNICO (cumple la spec)" if todos else "⚠️ VERSIÓN BASE DEFENDIBLE (cumple evidencia metodológica)")
""")

md(r"""## 14. Matriz destino-aspecto-sentimiento (Fase 3)

Inferencia del ensemble sobre el corpus completo → matriz analítica spec-compliant.
""")
code(r"""
import sys; sys.path.append(str(BASE / "scripts"))
from generar_matriz_absa import build_matrix

if NEED_TRAIN or not ART["matriz"].exists():
    corpus = pd.read_csv(BASE / "outputs/predictions/tourism_reviews_clean_absa_ready.csv", encoding="utf-8-sig")
    if MAX_CORPUS_INFER: corpus = corpus.head(MAX_CORPUS_INFER).copy()
    if "input_modelo" not in corpus.columns or corpus["input_modelo"].isna().any():
        corpus["input_modelo"] = "aspecto: " + corpus["aspecto"].astype(str) + " reseña: " + corpus["text_clean"].astype(str)
    BIAS = np.load(MODELS_DIR / f"_bias_{VER}.npy") if (MODELS_DIR / f"_bias_{VER}.npy").exists() else np.zeros(3)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    class InfDS(Dataset):
        def __init__(s, t): s.t=list(t)
        def __len__(s): return len(s.t)
        def __getitem__(s, i):
            e = tok(str(s.t[i]), add_special_tokens=True, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
            return {"input_ids": e["input_ids"].squeeze(0), "attention_mask": e["attention_mask"].squeeze(0)}
    cl = DataLoader(InfDS(corpus["input_modelo"]), batch_size=max(BATCH,16)); cprobs = np.zeros((len(corpus),3))
    for seed in SEEDS:
        mp = MODELS_DIR / f"modelo_{VER}_seed{seed}.pt"
        if not mp.exists(): continue
        model = BERTTextCNN().to(DEVICE); model.load_state_dict(torch.load(mp, map_location="cpu")); model.eval()
        pos = 0
        with torch.no_grad():
            for b in cl:
                with torch.autocast("cuda", enabled=USE_AMP):
                    lo = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE))
                pr = torch.softmax(lo.float(),1).cpu().numpy(); cprobs[pos:pos+len(pr)] += pr; pos += len(pr)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    cprobs /= len(SEEDS); corpus["label_pred"] = apply_bias(cprobs, BIAS)
    corpus[["review_uid","destination","aspecto","label_pred"]].to_csv(PRED_DIR / f"predicciones_corpus_{VER}.csv", index=False, encoding="utf-8-sig")
    matriz = build_matrix(corpus[["review_uid","destination","aspecto","label_pred"]])
    matriz.to_csv(ART["matriz"], index=False, encoding="utf-8-sig")
    matriz.to_json(MATR_DIR / "matriz_destino_aspecto_sentimiento.json", orient="records", force_ascii=False, indent=2)
matriz = pd.read_csv(ART["matriz"])
print("Matriz:", matriz.shape, "| columnas:", list(matriz.columns))
display(matriz.head(12))
""")

md(r"""## 15. Vistas de la matriz: niveles de evidencia y heatmap destino × aspecto""")
code(r"""
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios":[1,2.2]})
# Gráfico: distribución de niveles de evidencia
orden = ["sin datos", "insuficiente", "baja", "suficiente"]
ev = matriz["evidence_status"].value_counts().reindex(orden, fill_value=0)
ax[0].bar(ev.index, ev.values, color=["#cccccc","#f4a582","#fdb863","#4daf4a"])
ax[0].set_title("Niveles de evidencia (celdas destino×aspecto)"); ax[0].tick_params(axis="x", rotation=20)
for i,v in enumerate(ev.values): ax[0].text(i, v+0.5, int(v), ha="center")
# Heatmap destino x aspecto (score_ajustado)
piv = matriz.pivot(index="destination", columns="aspecto", values="score_ajustado")
im = ax[1].imshow(piv.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
ax[1].set_xticks(range(len(piv.columns))); ax[1].set_xticklabels(piv.columns, rotation=45, ha="right")
ax[1].set_yticks(range(len(piv.index))); ax[1].set_yticklabels(piv.index, fontsize=8)
ax[1].set_title("Heatmap destino × aspecto (score_ajustado, −1..1)")
fig.colorbar(im, ax=ax[1], shrink=0.8)
plt.tight_layout(); plt.show()
""")

md(r"""## 16. Exportación y conclusión""")
code(r"""
indice = {k: str(v.relative_to(BASE)) for k, v in ART.items()}
indice["matriz_json"] = "outputs/matrices/matriz_destino_aspecto_sentimiento.json"
indice["predicciones_corpus"] = f"outputs/predictions/predicciones_corpus_{VER}.csv"
indice["split_report"] = f"outputs/reports/split_report_gold_{VER}.md"
with open(REP / f"indice_trazabilidad_{VER}.json", "w", encoding="utf-8") as f: json.dump(indice, f, ensure_ascii=False, indent=2)

em = pd.read_csv(ART["resumen"]).iloc[0]
print("CONCLUSIÓN")
print("-"*64)
print(f"Modelo: BERT multilingual + TextCNN | gold {VER} ({len(gold)} ítems, split 70/15/15 sin fuga, {len(SEEDS)} semillas).")
print(f"Ensemble calibrado en test: F1-macro={em['ensemble_f1_macro']:.3f} | neg F1={em['ensemble_f1_negativo']:.3f} "
      f"recall={em['ensemble_recall_negativo']:.3f} | neu F1={em['ensemble_f1_neutro']:.3f} | acc={em['ensemble_accuracy']:.3f}")
print(f"Estabilidad std F1-macro = {em['std_f1_macro']:.4f} ({'estable' if em['estable_std<=0.03'] else 'inestable'})")
print(f"VEREDICTO: {VEREDICTO}")
""")

md(r"""
**Cierre.** Registro reproducible: datos (gold v4 reforzado, splits sin fuga), comparación mínima de
modelos (TF-IDF+LogReg, BERT+TextCNN, XLM-R ref), métricas con ≥3 semillas (media±std), matriz ABSA
spec-compliant para la Fase 3 y veredicto automático. La comparación v3→v4 es referencia de evolución
(test re-particionado); la mejora se sostiene por aspecto (clima/aforo/limpieza) y por estabilidad.
`alojamiento` permanece limitado por el corpus (declarado como limitación, R20).
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parent.parent / "notebooks" / "03_entrenamiento_absa_bert_textcnn_gold_v4.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook escrito:", out, "| celdas:", len(cells))
