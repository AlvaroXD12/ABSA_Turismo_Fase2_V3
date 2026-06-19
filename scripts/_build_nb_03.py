# -*- coding: utf-8 -*-
# Construye el notebook 03: bitacora metodologica AUTONOMA que ENTRENA al
# ejecutarse de principio a fin (para correr en otra PC mas potente).
import json
from pathlib import Path

cells = []
def md(t):  cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})

md(r"""
# 03 — Entrenamiento ABSA (BERT multilingual + TextCNN) sobre gold set v3

**Bitácora metodológica reproducible.** Este notebook entrena y evalúa el módulo ABSA
de la Fase 2 sobre el **gold set v3** (ampliado priorizando la clase negativa) y deja
registro visible de: qué datos se usaron, qué configuración se entrenó, qué resultó y
si cumple la especificación.

- **Especificación:** [`specs/modulo-absa-fase2.md`](../specs/modulo-absa-fase2.md)
- **Partición:** `outputs/reports/split_report_gold_v3.md`

**Objetivo (criterio de éxito):** F1-macro ≥ 0.70 en test, con mínimos por clase
(negativo F1 ≥ 0.60 y recall ≥ 0.60; neutro F1 ≥ 0.60). Si no se alcanza, se entrega
como *versión base defendible* con comparación, análisis de errores y límites del gold.

### Cómo ejecutarlo
1. Crear entorno e instalar dependencias: `pip install -r requirements.txt`
   (instala PyTorch acorde a tu CUDA primero — ver `requirements.txt`).
2. Ejecutar todas las celdas en orden (Kernel → Restart & Run All).
3. **El notebook ENTRENA** (3 semillas). En GPU potente toma decenas de minutos; en
   GPU modesta, varias horas. Ajusta `BATCH` y `USE_GRADIENT_CHECKPOINTING` abajo.
""")

code(r"""
from pathlib import Path
import json, random, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import (precision_recall_fscore_support, accuracy_score,
                             confusion_matrix, classification_report)

pd.set_option("display.max_columns", 60); pd.set_option("display.width", 200)

# Raiz del proyecto (el notebook vive en notebooks/)
BASE = Path.cwd().parent if Path.cwd().name.lower() == "notebooks" else Path.cwd()
DATA, REP, VIS = BASE / "data", BASE / "outputs" / "reports", BASE / "outputs" / "visualizations"
MODELS_DIR, PRED_DIR, MATR_DIR = BASE / "models", BASE / "outputs" / "predictions", BASE / "outputs" / "matrices"
for d in (REP, VIS, MODELS_DIR, PRED_DIR, MATR_DIR): d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
LABELS = ["negativo", "neutro", "positivo"]; L2I = {l: i for i, l in enumerate(LABELS)}; I2L = {i: l for l, i in L2I.items()}

# ---------------- Config de entrenamiento ----------------
MODEL_NAME = "bert-base-multilingual-cased"
MAX_LEN, BATCH, EPOCHS = 256, 8, 12
LR, WEIGHT_DECAY, WARMUP_RATIO, PATIENCE, DROPOUT = 2e-5, 0.10, 0.10, 3, 0.40
CNN_FILTERS, CNN_KERNELS = 128, (2, 3, 4)
SEEDS = [42, 7, 123, 2024, 77]  # 5 semillas: mejor ensemble + menos varianza (apunta a 0.70)
FOCAL_GAMMA, NEG_BOOST = 1.0, 1.2   # menos agresivo: recupera precision del negativo
LABEL_SMOOTHING = 0.1               # mejor calibracion de probabilidades (ayuda neg/neutro)
CALIBRAR_DECISION = True            # tras entrenar, ajusta la frontera en val para maximizar F1-macro
# GPU con poca VRAM (<6GB): pon True y baja BATCH a 4. GPU potente: deja False y sube BATCH.
USE_GRADIENT_CHECKPOINTING = False
# Umbrales de la spec
TH_MACRO, TH_NEG_F1, TH_NEG_REC, TH_NEU_F1 = 0.70, 0.60, 0.60, 0.60

print("BASE:", BASE, "| DEVICE:", DEVICE, "| AMP:", USE_AMP)
print("Config:", dict(model=MODEL_NAME, max_len=MAX_LEN, batch=BATCH, epochs=EPOCHS, lr=LR,
                      seeds=SEEDS, focal_gamma=FOCAL_GAMMA, neg_boost=NEG_BOOST,
                      grad_checkpoint=USE_GRADIENT_CHECKPOINTING))
""")

md(r"""
## 1. Carga del gold set v3 y de los splits

Gold v3 (3.562 ítems aspecto-texto-label, consenso de 3 anotadores, κ≈0.89) y las
particiones `train/val/test_gold_v3.csv` (estratificadas por polaridad, agrupadas por
`review_uid`).
""")

code(r"""
gold  = pd.read_csv(DATA / "gold_set_v3.csv", encoding="utf-8-sig")
train = pd.read_csv(DATA / "train_gold_v3.csv", encoding="utf-8-sig")
val   = pd.read_csv(DATA / "val_gold_v3.csv", encoding="utf-8-sig")
test  = pd.read_csv(DATA / "test_gold_v3.csv", encoding="utf-8-sig")
for d in (gold, train, val, test):
    d["label"] = d["label"].astype(str).str.lower().str.strip()
    if "input_modelo" not in d.columns or d["input_modelo"].isna().any():
        d["input_modelo"] = "aspecto: " + d["aspecto"].astype(str) + " reseña: " + d["text_clean"].astype(str)

print(f"gold v3 : {len(gold):5d} ítems | {gold['review_uid'].nunique()} reseñas únicas")
for nm, d in [("train", train), ("val", val), ("test", test)]:
    print(f"{nm:5s}   : {len(d):5d} ({len(d)/len(gold)*100:.1f}%)")
train.head(3)
""")

md(r"""
## 2. Verificación de no fuga por `review_uid`

Ninguna reseña puede aparecer en más de un split.
""")

code(r"""
s_tr, s_va, s_te = set(train.review_uid), set(val.review_uid), set(test.review_uid)
fuga = {"train∩val": len(s_tr & s_va), "train∩test": len(s_tr & s_te), "val∩test": len(s_va & s_te)}
print("Solapamientos de review_uid:", fuga)
assert sum(fuga.values()) == 0, "FUGA detectada entre splits"
print("✅ Sin fuga: cada reseña aparece en un solo split.")
""")

md(r"""
## 3. Distribución por split (polaridad, aspecto, destino, idioma)
""")

code(r"""
alld = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
def tabla(col):
    # groupby+size evita el conflicto de usar la misma columna como index y values
    t = alld.groupby([col, "split"]).size().unstack("split", fill_value=0)
    return t.reindex(columns=["train", "val", "test"], fill_value=0)

print("Polaridad por split (conteo):"); display(tabla("label"))
print("Polaridad por split (%):");      display((tabla("label") / tabla("label").sum() * 100).round(1))
print("Aspecto por split:");            display(tabla("aspecto"))
if "language_review" in train: print("Idioma por split:"); display(tabla("language_review"))
if "destination" in train:     print("Destino por split:"); display(tabla("destination"))

prop = (tabla("label") / tabla("label").sum() * 100)[["train", "val", "test"]]
ax = prop.T.plot(kind="bar", figsize=(7, 4)); ax.set_ylabel("% ítems")
ax.set_title("Distribución de polaridad por split (gold v3)"); plt.xticks(rotation=0)
plt.tight_layout(); plt.show()
""")

md(r"""
## 4. Modelo: BERT multilingual + TextCNN

Encoder Transformer multilingüe → convoluciones 1D (kernels 2/3/4) + max-pooling →
clasificador lineal de 3 clases. La ablación previa mostró que cambiar el encoder no es
la palanca; se conserva esta arquitectura base y se ataca la señal de la clase negativa.
""")

code(r"""
class ABSADataset(Dataset):
    def __init__(self, texts, labels, tok):
        self.t = list(texts); self.l = list(labels); self.tok = tok
    def __len__(self): return len(self.t)
    def __getitem__(self, i):
        e = self.tok(str(self.t[i]), add_special_tokens=True, max_length=MAX_LEN, padding="max_length",
                     truncation=True, return_attention_mask=True, return_tensors="pt")
        return {"input_ids": e["input_ids"].squeeze(0), "attention_mask": e["attention_mask"].squeeze(0),
                "labels": torch.tensor(L2I[self.l[i]], dtype=torch.long)}

class BERTTextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        if USE_GRADIENT_CHECKPOINTING:
            self.bert.config.use_cache = False
            self.bert.gradient_checkpointing_enable()
        h = self.bert.config.hidden_size
        self.convs = nn.ModuleList([nn.Conv1d(h, CNN_FILTERS, k) for k in CNN_KERNELS])
        self.drop = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(CNN_FILTERS * len(CNN_KERNELS), 3)
    def forward(self, ids, mask):
        x = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state.transpose(1, 2)
        pooled = [torch.max(torch.relu(c(x)), dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(pooled, dim=1)))

print("Modelo definido:", BERTTextCNN.__name__)
""")

md(r"""
## 5. Configuración enfocada a la clase negativa

La clase negativa es el cuello de botella. Se combinan **class weights** inversos a la
frecuencia con **refuerzo extra a negativa** (`NEG_BOOST=1.5`) y **Focal Loss**
(`gamma=2.0`), que enfoca los ejemplos difíciles (típicamente negativos confundidos con
neutro). Abajo se muestran los pesos reales por clase.
""")

code(r"""
class FocalLoss(nn.Module):
    def __init__(self, weight, gamma): super().__init__(); self.w=weight; self.g=gamma
    def forward(self, logits, y):
        ce = nn.functional.cross_entropy(logits, y, weight=self.w, reduction="none",
                                         label_smoothing=LABEL_SMOOTHING)
        return (((1 - torch.exp(-ce)) ** self.g) * ce).mean()

def class_weights(labels):
    c = pd.Series(labels).value_counts().reindex(LABELS, fill_value=0); tot = c.sum()
    w = [tot / (len(LABELS) * c[l]) if c[l] > 0 else 0.0 for l in LABELS]
    w[L2I["negativo"]] *= NEG_BOOST
    return torch.tensor(w, dtype=torch.float).to(DEVICE)

_w = class_weights(train["label"])
print("Pesos por clase (con refuerzo a negativa):")
for l in LABELS: print(f"  {l:9s}: {float(_w[L2I[l]]):.3f}")
print(f"Focal gamma = {FOCAL_GAMMA} | NEG_BOOST = {NEG_BOOST}")
""")

md(r"""
## 6. Entrenamiento con 5 semillas + calibración de decisión

Cada semilla: entrenamiento con early stopping por F1-macro en validación; se guarda el
mejor checkpoint. Al final, **ensemble** por promedio de probabilidades de las 5 semillas.

Para cerrar el último tramo hacia 0.70 sin re-entrenar, se añade **calibración de la
frontera de decisión**: sobre el conjunto de **validación** se busca el sesgo por clase
(en log-probabilidad) que maximiza el F1-macro, y ese mismo ajuste se aplica a test y al
corpus. Esto **rebalancea la precisión/recall de la clase negativa** (que el modelo tendía
a sobre-predecir). Es una técnica estándar: se ajusta en val, se evalúa en test.
""")

code(r"""
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def metrics(trues, preds):
    pr, rc, f1, _ = precision_recall_fscore_support(trues, preds, labels=LABELS, average=None, zero_division=0)
    _, _, mf1, _ = precision_recall_fscore_support(trues, preds, labels=LABELS, average="macro", zero_division=0)
    out = {"f1_macro": mf1, "accuracy": accuracy_score(trues, preds)}
    for i, l in enumerate(LABELS): out[f"f1_{l}"] = f1[i]; out[f"recall_{l}"] = rc[i]
    return out

def predict(model, loader):
    model.eval(); P, T = [], []
    with torch.no_grad():
        for b in loader:
            with torch.autocast("cuda", enabled=USE_AMP):
                lo = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE))
            P.append(torch.softmax(lo.float(), 1).cpu().numpy()); T += [I2L[y] for y in b["labels"].numpy()]
    return np.concatenate(P), T

def apply_bias(probs, bias):
    return [I2L[i] for i in (np.log(probs + 1e-9) + bias).argmax(1)]

def best_bias(val_probs, val_trues):
    # Calibracion de decision: busca el sesgo por clase (en log-prob) que maximiza
    # F1-macro en VALIDACION. Rebalancea precision/recall del negativo sin re-entrenar.
    # Se fija el sesgo de 'positivo' en 0 (solo importan las diferencias).
    if not CALIBRAR_DECISION:
        return np.zeros(3)
    logp = np.log(val_probs + 1e-9)
    grid = np.arange(-1.2, 1.21, 0.2)
    best, bb = -1.0, np.zeros(3)
    for b0 in grid:
        for b1 in grid:
            b = np.array([b0, b1, 0.0])
            f1 = metrics(val_trues, [I2L[i] for i in (logp + b).argmax(1)])["f1_macro"]
            if f1 > best: best, bb = f1, b
    return bb

def train_one(seed):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tl = DataLoader(ABSADataset(train["input_modelo"], train["label"], tok), batch_size=BATCH, shuffle=True)
    vl = DataLoader(ABSADataset(val["input_modelo"], val["label"], tok), batch_size=BATCH)
    el = DataLoader(ABSADataset(test["input_modelo"], test["label"], tok), batch_size=BATCH)
    model = BERTTextCNN().to(DEVICE)
    loss_fn = FocalLoss(class_weights(train["label"]), FOCAL_GAMMA)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    tot = len(tl) * EPOCHS
    sch = get_linear_schedule_with_warmup(opt, int(tot * WARMUP_RATIO), tot)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    best, best_state, pat = -1, None, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for b in tl:
            opt.zero_grad()
            with torch.autocast("cuda", enabled=USE_AMP):
                loss = loss_fn(model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE)), b["labels"].to(DEVICE))
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
        vp, vt = predict(model, vl); vf = metrics(vt, [I2L[i] for i in vp.argmax(1)])["f1_macro"]
        print(f"    seed {seed} epoch {ep:2d}/{EPOCHS}  val_f1_macro={vf:.3f}")
        if vf > best: best, best_state, pat = vf, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= PATIENCE: print("    early stopping"); break
    if best_state: model.load_state_dict(best_state)
    torch.save(best_state, MODELS_DIR / f"modelo_v3_seed{seed}.pt")   # para inferencia sobre el corpus
    tp, tt = predict(model, el)
    vp, vtv = predict(model, vl)          # probs de val (para calibrar la decision)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return tp, tt, vp, vtv

t0 = time.time()
rows, probs, val_probs, test_trues, val_trues = [], [], [], None, None
for seed in SEEDS:
    print(f"=== Semilla {seed} ===")
    p, tt, vp, vtv = train_one(seed); test_trues = tt; val_trues = vtv; probs.append(p); val_probs.append(vp)
    m = metrics(tt, [I2L[i] for i in p.argmax(1)])
    rows.append({"seed": seed, **{k: round(v, 4) for k, v in m.items()}})
    print(f"  -> f1_macro={m['f1_macro']:.3f} f1_neg={m['f1_negativo']:.3f} rec_neg={m['recall_negativo']:.3f} f1_neu={m['f1_neutro']:.3f}")

det = pd.DataFrame(rows)
ens_probs = np.mean(probs, axis=0)        # ensemble en test
ens_val   = np.mean(val_probs, axis=0)    # ensemble en val (para calibrar)

# Calibracion de la frontera de decision sobre validacion -> se aplica a test
BIAS = best_bias(ens_val, val_trues)
ens_preds_raw = [I2L[i] for i in ens_probs.argmax(1)]
ens_preds     = apply_bias(ens_probs, BIAS)          # decision calibrada (oficial)
ens_metrics   = metrics(test_trues, ens_preds)
f1_raw  = metrics(test_trues, ens_preds_raw)["f1_macro"]
det.to_csv(REP / "resultados_bert_textcnn_v3.csv", index=False, encoding="utf-8-sig")
print(f"\nEntrenamiento completo en {(time.time()-t0)/60:.1f} min")
print(f"Sesgo de calibracion (neg,neu,pos): {np.round(BIAS,2)}")
print(f"F1-macro ensemble: SIN calibrar={f1_raw:.4f}  ->  CON calibracion={ens_metrics['f1_macro']:.4f}")
""")

md(r"""## 7. Resultados por semilla""")
code(r"""display(det)""")

md(r"""## 8. Promedio ± desviación estándar

La selección no se basa en la mejor corrida sino en el promedio. Std del F1-macro > 0.03
indica inestabilidad.
""")
code(r"""
cols = [c for c in det.columns if c != "seed"]
agg = pd.DataFrame({"media": det[cols].mean().round(4), "desv_std": det[cols].std().round(4)})
display(agg)
std_macro = det["f1_macro"].std()
print(f"Std F1-macro = {std_macro:.4f} -> {'ESTABLE' if std_macro <= 0.03 else 'INESTABLE (>0.03)'}")
""")

md(r"""## 9. Métricas obligatorias (ensemble): F1-macro, por clase, recall negativo, matriz de confusión, por aspecto""")
code(r"""
print("Classification report (ensemble):")
cr = pd.DataFrame(classification_report(test_trues, ens_preds, labels=LABELS, output_dict=True, zero_division=0)).T
display(cr.round(3)); cr.to_csv(REP / "classification_report_v3.csv", encoding="utf-8-sig")

# Por aspecto
ta = test.copy(); ta["pred"] = ens_preds; filas = []
for asp, g in ta.groupby("aspecto"):
    _, _, f1, _ = precision_recall_fscore_support(g["label"], g["pred"], labels=LABELS, average="macro", zero_division=0)
    filas.append({"aspecto": asp, "soporte": len(g), "f1_macro": round(f1, 4)})
por_asp = pd.DataFrame(filas).sort_values("f1_macro"); por_asp.to_csv(REP / "por_aspecto_v3.csv", index=False, encoding="utf-8-sig")
print("F1-macro por aspecto (peor a mejor):"); display(por_asp)

cm = confusion_matrix(test_trues, ens_preds, labels=LABELS)
plt.figure(figsize=(5, 4)); plt.imshow(cm, cmap="Blues"); plt.colorbar()
plt.xticks(range(3), LABELS, rotation=45); plt.yticks(range(3), LABELS)
for i in range(3):
    for j in range(3): plt.text(j, i, cm[i, j], ha="center", va="center")
plt.title("Matriz de confusión (ensemble)"); plt.ylabel("real"); plt.xlabel("pred")
plt.tight_layout(); plt.savefig(VIS / "matriz_confusion_v3.png", dpi=180); plt.show()
""")

md(r"""
## 10. Veredicto automático contra la especificación

- **Éxito técnico:** F1-macro ≥ 0.70 **y** negativo F1 ≥ 0.60 **y** recall negativo ≥ 0.60 **y** neutro F1 ≥ 0.60.
- **Versión base defendible:** si no, pero con comparación, análisis de errores y límites del gold.
""")
code(r"""
checks = {
    "F1-macro ≥ 0.70":        (ens_metrics["f1_macro"], TH_MACRO),
    "Negativo F1 ≥ 0.60":     (ens_metrics["f1_negativo"], TH_NEG_F1),
    "Recall negativo ≥ 0.60": (ens_metrics["recall_negativo"], TH_NEG_REC),
    "Neutro F1 ≥ 0.60":       (ens_metrics["f1_neutro"], TH_NEU_F1),
}
print("=== Veredicto contra la spec (ensemble) ===")
todos = True
for k, (v, th) in checks.items():
    ok = v >= th; todos &= ok
    print(f"  [{'PASA' if ok else 'FALLA'}] {k:24s} -> {v:.3f}")
estable = det["f1_macro"].std() <= 0.03
print(f"  [{'OK' if estable else 'X'}] Estabilidad std F1-macro ≤ 0.03 -> {det['f1_macro'].std():.4f}")
VEREDICTO = "ÉXITO TÉCNICO" if todos else "VERSIÓN BASE DEFENDIBLE"
print("\nRESULTADO:", "✅ ÉXITO TÉCNICO (cumple la spec)" if todos
      else "⚠️ NO alcanza el umbral -> VERSIÓN BASE DEFENDIBLE (con evidencia metodológica)")
""")

md(r"""
## 11. Inferencia sobre el corpus completo

Para construir la matriz que consume la Fase 3 se aplica el **ensemble** de los modelos
entrenados (uno por semilla) sobre **todo el corpus** (`tourism_reviews_clean_absa_ready.csv`,
≈21.040 pares reseña×aspecto). Se promedian las probabilidades de las semillas y se guarda
`outputs/predictions/predicciones_corpus_v3.csv` (`review_uid, destination, aspecto, label_pred`).
Así las predicciones viajan por git (no hace falta transferir el modelo `.pt`).
""")
code(r"""
corpus = pd.read_csv(BASE / "outputs/predictions/tourism_reviews_clean_absa_ready.csv", encoding="utf-8-sig")
if "input_modelo" not in corpus.columns or corpus["input_modelo"].isna().any():
    corpus["input_modelo"] = "aspecto: " + corpus["aspecto"].astype(str) + " reseña: " + corpus["text_clean"].astype(str)
print("Corpus a predecir:", len(corpus), "pares reseña×aspecto")

class InferDS(Dataset):
    def __init__(self, texts, tok): self.t=list(texts); self.tok=tok
    def __len__(self): return len(self.t)
    def __getitem__(self, i):
        e = self.tok(str(self.t[i]), add_special_tokens=True, max_length=MAX_LEN, padding="max_length",
                     truncation=True, return_attention_mask=True, return_tensors="pt")
        return {"input_ids": e["input_ids"].squeeze(0), "attention_mask": e["attention_mask"].squeeze(0)}

_tok = AutoTokenizer.from_pretrained(MODEL_NAME)
cl = DataLoader(InferDS(corpus["input_modelo"], _tok), batch_size=max(BATCH, 16))
corpus_probs = np.zeros((len(corpus), 3))
for seed in SEEDS:                                  # ensemble: una semilla a la vez (poca memoria)
    model = BERTTextCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODELS_DIR / f"modelo_v3_seed{seed}.pt", map_location="cpu"))
    model.eval()
    pos = 0
    with torch.no_grad():
        for b in cl:
            with torch.autocast("cuda", enabled=USE_AMP):
                lo = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE))
            p = torch.softmax(lo.float(), 1).cpu().numpy()
            corpus_probs[pos:pos+len(p)] += p; pos += len(p)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"  inferencia corpus seed {seed} lista")
corpus_probs /= len(SEEDS)
corpus["label_pred"] = apply_bias(corpus_probs, BIAS)   # misma decision calibrada que en test
pred_cols = ["review_uid", "destination", "aspecto", "label_pred"]
corpus[pred_cols].to_csv(PRED_DIR / "predicciones_corpus_v3.csv", index=False, encoding="utf-8-sig")
print("Predicciones corpus ->", PRED_DIR / "predicciones_corpus_v3.csv")
print("Distribución de polaridad predicha en el corpus:", corpus["label_pred"].value_counts().to_dict())
""")

md(r"""
## 12. Matriz destino-aspecto-sentimiento

Con las predicciones del modelo sobre el corpus se construye la matriz analítica que
consumirá la Fase 3 (campos por celda, score, confianza, niveles de evidencia y conflicto;
reglas R12–R16 de la spec). Se reutiliza `scripts/generar_matriz_absa.py`.
""")
code(r"""
import sys
sys.path.append(str(BASE / "scripts"))
from generar_matriz_absa import build_matrix

matriz = build_matrix(corpus[["review_uid", "destination", "aspecto", "label_pred"]])
matriz.to_csv(MATR_DIR / "matriz_destino_aspecto_sentimiento.csv", index=False, encoding="utf-8-sig")
print("Matriz ->", MATR_DIR / "matriz_destino_aspecto_sentimiento.csv", "| celdas:", len(matriz))
print("Niveles de evidencia:", matriz["nivel_evidencia"].value_counts().to_dict())
print("Celdas con conflicto:", int(matriz["conflict_flag"].sum()))
display(matriz.head(20))
""")

md(r"""## 13. Exportación de resultados (trazabilidad)""")
code(r"""
resumen = {**{f"ensemble_{k}": round(v, 4) for k, v in ens_metrics.items()},
           **{f"media_{c}": round(det[c].mean(), 4) for c in cols},
           **{f"std_{c}": round(det[c].std(), 4) for c in cols},
           "veredicto": VEREDICTO, "estable_std<=0.03": bool(det['f1_macro'].std() <= 0.03)}
pd.DataFrame([resumen]).to_csv(REP / "resumen_bert_textcnn_v3.csv", index=False, encoding="utf-8-sig")
indice = {
    "resultados_por_semilla": "outputs/reports/resultados_bert_textcnn_v3.csv",
    "resumen_y_veredicto": "outputs/reports/resumen_bert_textcnn_v3.csv",
    "classification_report": "outputs/reports/classification_report_v3.csv",
    "por_aspecto": "outputs/reports/por_aspecto_v3.csv",
    "matriz_confusion": "outputs/visualizations/matriz_confusion_v3.png",
    "predicciones_corpus": "outputs/predictions/predicciones_corpus_v3.csv",
    "matriz_destino_aspecto": "outputs/matrices/matriz_destino_aspecto_sentimiento.csv",
    "splits": ["data/train_gold_v3.csv", "data/val_gold_v3.csv", "data/test_gold_v3.csv"],
}
with open(REP / "indice_trazabilidad_v3.json", "w", encoding="utf-8") as f:
    json.dump(indice, f, ensure_ascii=False, indent=2)
print("Artefactos guardados en outputs/. Índice -> outputs/reports/indice_trazabilidad_v3.json")
for k, v in indice.items(): print(f"  {k}: {v}")
""")

md(r"""## 14. Conclusión""")
code(r"""
em = ens_metrics
print("CONCLUSIÓN (generada de los resultados)")
print("-" * 60)
print(f"Modelo: BERT multilingual + TextCNN | gold v3 (3.562 ítems, split 70/15/15 sin fuga).")
print(f"Config enfocada a negativa: focal γ={FOCAL_GAMMA}, class weights con NEG_BOOST={NEG_BOOST}, {len(SEEDS)} semillas.")
print(f"Ensemble en test -> F1-macro={em['f1_macro']:.3f} | neg F1={em['f1_negativo']:.3f} "
      f"recall={em['recall_negativo']:.3f} | neu F1={em['f1_neutro']:.3f} | acc={em['accuracy']:.3f}")
print(f"Estabilidad (std F1-macro entre semillas): {det['f1_macro'].std():.4f}")
print(f"VEREDICTO: {VEREDICTO}")
if VEREDICTO != "ÉXITO TÉCNICO":
    print("\nComo versión base defendible, se aporta: comparación por semilla, matriz de")
    print("confusión, desempeño por clase y por aspecto, y límites del gold (p. ej. escasez")
    print("intrínseca de negativos en alojamiento/gastronomia, limitada por el corpus).")
""")

md(r"""
**Cierre metodológico.** Este notebook deja registro reproducible de todo el proceso:
datos (gold v3 y splits sin fuga), configuración (BERT+TextCNN enfocado a la negativa),
resultados con ≥3 semillas (media ± desviación) y veredicto automático contra la spec.
Si el ensemble no alcanza F1-macro ≥ 0.70, el módulo se reporta como versión base
defendible con toda la evidencia anterior, no como modelo definitivo.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parent.parent / "notebooks" / "03_entrenamiento_absa_bert_textcnn_gold_v3.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook escrito:", out, "| celdas:", len(cells))
