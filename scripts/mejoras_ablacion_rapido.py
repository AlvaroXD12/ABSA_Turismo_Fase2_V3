# -*- coding: utf-8 -*-
# ============================================================
# Ablacion de mejoras para subir el F1-macro (NO toca el notebook)
# ------------------------------------------------------------
# Compara, sobre los splits reales train/val/test, varias palancas:
#   1) Encoder:  mBERT  vs  XLM-RoBERTa  vs  XLM-R preentrenado en sentimiento
#   2) Cabezal:  TextCNN (el actual)     vs  fine-tuning simple ([CLS] + lineal)
#   3) Ensemble: promedio de probabilidades sobre 3 semillas
# Reporta F1-macro (media +/- std de semillas), F1-macro del ENSEMBLE y,
# en especial, el F1 de la clase NEUTRO (la palanca aritmetica clave).
#
# Tecnicas para entrar en 4GB de GPU y ayudar con datos limitados:
#   - Precision mixta (AMP)
#   - Congelado de embeddings + primeras N capas del encoder
#
# Salidas:
#   outputs/reports/mejoras_ablacion_rapido_detalle.csv
#   outputs/reports/mejoras_ablacion_rapido_resumen.csv
#   outputs/visualizations/mejoras_ablacion_rapido_f1.png
# ============================================================

from pathlib import Path
import random, traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
try:
    from transformers import get_linear_schedule_with_warmup
except Exception:
    from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"
VIS_DIR = BASE_DIR / "outputs" / "visualizations"
for d in [REPORTS_DIR, VIS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

VALID_LABELS = ["negativo", "neutro", "positivo"]
LABEL_TO_ID = {l: i for i, l in enumerate(VALID_LABELS)}
ID_TO_LABEL = {i: l for l, i in LABEL_TO_ID.items()}

# Hiperparametros (VERSION RAPIDA: menos epocas/seeds/longitud -> resultado en ~1-1.5h)
MAX_LEN = 160
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2.5e-5
WEIGHT_DECAY = 0.10
WARMUP_RATIO = 0.10
PATIENCE = 2
DROPOUT = 0.40
FREEZE_BOTTOM = 6          # congela embeddings + primeras 6 capas (memoria + datos limitados)
SEEDS = [42, 7]            # 2 semillas -> media+/-std y ensemble de 2

# Configuraciones clave (las 3 mas informativas): (nombre, modelo_hf, tipo_cabezal)
CONFIGS = [
    ("mbert_textcnn",     "bert-base-multilingual-cased",                 "cnn"),     # ~tu modelo actual (referencia)
    ("xlmr_simple",       "xlm-roberta-base",                             "simple"),  # mejor encoder + cabezal simple
    ("xlmr_sent_simple",  "cardiffnlp/twitter-xlm-roberta-base-sentiment","simple"),  # preentrenado en sentimiento
]

print("DEVICE:", DEVICE, "| AMP:", USE_AMP)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_split(name):
    df = pd.read_csv(DATA_DIR / f"{name}.csv", encoding="utf-8-sig")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    if "input_modelo" not in df.columns or df["input_modelo"].isna().any():
        df["input_modelo"] = "aspecto: " + df["aspecto"].astype(str) + " reseña: " + df["text_clean"].astype(str)
    return df[df["label"].isin(VALID_LABELS)].reset_index(drop=True)


train_df = load_split("train")
val_df = load_split("validation")
test_df = load_split("test")
print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")


class ABSADataset(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.texts = list(texts); self.labels = list(labels); self.tok = tok; self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        e = self.tok(str(self.texts[i]), add_special_tokens=True, max_length=self.max_len,
                     padding="max_length", truncation=True, return_attention_mask=True, return_tensors="pt")
        return {"input_ids": e["input_ids"].squeeze(0),
                "attention_mask": e["attention_mask"].squeeze(0),
                "labels": torch.tensor(LABEL_TO_ID[self.labels[i]], dtype=torch.long)}


def freeze_bottom(encoder, n):
    """Congela embeddings + primeras n capas (compatible con BERT y RoBERTa)."""
    if hasattr(encoder, "embeddings"):
        for p in encoder.embeddings.parameters():
            p.requires_grad = False
    layers = getattr(getattr(encoder, "encoder", None), "layer", None)
    if layers is not None:
        for layer in layers[:n]:
            for p in layer.parameters():
                p.requires_grad = False


class ABSAModel(nn.Module):
    def __init__(self, model_name, head, num_labels, dropout):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        freeze_bottom(self.encoder, FREEZE_BOTTOM)
        h = self.encoder.config.hidden_size
        self.head = head
        self.dropout = nn.Dropout(dropout)
        if head == "cnn":
            self.convs = nn.ModuleList([nn.Conv1d(h, 128, k) for k in (2, 3, 4)])
            self.classifier = nn.Linear(128 * 3, num_labels)
        else:
            self.classifier = nn.Linear(h, num_labels)

    def forward(self, input_ids, attention_mask):
        seq = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.head == "cnn":
            x = seq.transpose(1, 2)
            pooled = [torch.max(torch.relu(conv(x)), dim=2).values for conv in self.convs]
            x = torch.cat(pooled, dim=1)
        else:
            x = seq[:, 0]   # [CLS]/<s>
        return self.classifier(self.dropout(x))


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.0):
        super().__init__(); self.weight = weight; self.gamma = gamma
    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def class_weights(labels):
    c = pd.Series(labels).value_counts().reindex(VALID_LABELS, fill_value=0)
    tot = c.sum()
    return torch.tensor([tot / (len(VALID_LABELS) * c[l]) if c[l] > 0 else 0.0 for l in VALID_LABELS],
                        dtype=torch.float).to(DEVICE)


def predict_probs(model, loader):
    model.eval()
    probs, trues = [], []
    with torch.no_grad():
        for b in loader:
            ids = b["input_ids"].to(DEVICE); m = b["attention_mask"].to(DEVICE)
            with torch.autocast("cuda", enabled=USE_AMP):
                logits = model(ids, m)
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            trues += [ID_TO_LABEL[y] for y in b["labels"].numpy()]
    return np.concatenate(probs), trues


def macro_f1(trues, preds):
    _, _, f1, _ = precision_recall_fscore_support(trues, preds, labels=VALID_LABELS, average="macro", zero_division=0)
    return f1


def per_class_f1(trues, preds):
    _, _, f1, _ = precision_recall_fscore_support(trues, preds, labels=VALID_LABELS, average=None, zero_division=0)
    return dict(zip(VALID_LABELS, f1))


def train_one(model_name, head, seed, batch_size):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    tl = DataLoader(ABSADataset(train_df["input_modelo"], train_df["label"], tok, MAX_LEN), batch_size=batch_size, shuffle=True)
    vl = DataLoader(ABSADataset(val_df["input_modelo"], val_df["label"], tok, MAX_LEN), batch_size=batch_size)
    el = DataLoader(ABSADataset(test_df["input_modelo"], test_df["label"], tok, MAX_LEN), batch_size=batch_size)

    model = ABSAModel(model_name, head, len(VALID_LABELS), DROPOUT).to(DEVICE)
    loss_fn = FocalLoss(weight=class_weights(train_df["label"]), gamma=0.0)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total = len(tl) * EPOCHS
    sched = get_linear_schedule_with_warmup(optim, int(total * WARMUP_RATIO), total)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_val, best_state, patience = -1.0, None, 0
    for _ in range(EPOCHS):
        model.train()
        for b in tl:
            ids = b["input_ids"].to(DEVICE); m = b["attention_mask"].to(DEVICE); y = b["labels"].to(DEVICE)
            optim.zero_grad()
            with torch.autocast("cuda", enabled=USE_AMP):
                loss = loss_fn(model(ids, m), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optim); scaler.update(); sched.step()
        vp, vt = predict_probs(model, vl)
        vf1 = macro_f1(vt, [ID_TO_LABEL[i] for i in vp.argmax(1)])
        if vf1 > best_val:
            best_val = vf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_probs, test_trues = predict_probs(model, el)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return test_probs, test_trues


# ------------------------------------------------------------
rows = []
resumen = []
ensemble_f1_by_config = {}

for name, model_name, head in CONFIGS:
    print(f"\n=== {name}  ({model_name}, head={head}) ===")
    seed_probs = []
    test_trues = None
    bs = BATCH_SIZE
    for seed in SEEDS:
        try:
            probs, trues = train_one(model_name, head, seed, bs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bs > 2:
                torch.cuda.empty_cache(); bs = max(2, bs // 2)
                print(f"  OOM -> reintento con batch_size={bs}")
                probs, trues = train_one(model_name, head, seed, bs)
            else:
                print("  ERROR:", e); traceback.print_exc(); continue
        test_trues = trues
        seed_probs.append(probs)
        preds = [ID_TO_LABEL[i] for i in probs.argmax(1)]
        f1 = macro_f1(trues, preds); pc = per_class_f1(trues, preds)
        acc = accuracy_score(trues, preds)
        rows.append({"config": name, "modelo": model_name, "head": head, "seed": seed,
                     "f1_macro": round(f1, 4), "accuracy": round(acc, 4),
                     "f1_negativo": round(pc["negativo"], 4), "f1_neutro": round(pc["neutro"], 4),
                     "f1_positivo": round(pc["positivo"], 4)})
        print(f"  seed={seed:3d} f1_macro={f1:.3f} acc={acc:.3f} f1_neutro={pc['neutro']:.3f}")
        pd.DataFrame(rows).to_csv(REPORTS_DIR / "mejoras_ablacion_rapido_detalle.csv", index=False, encoding="utf-8-sig")

    if not seed_probs:
        continue
    # Ensemble: promedio de probabilidades sobre semillas
    ens_probs = np.mean(seed_probs, axis=0)
    ens_preds = [ID_TO_LABEL[i] for i in ens_probs.argmax(1)]
    ens_f1 = macro_f1(test_trues, ens_preds); ens_pc = per_class_f1(test_trues, ens_preds)
    ens_acc = accuracy_score(test_trues, ens_preds)
    ensemble_f1_by_config[name] = ens_f1
    cfg_rows = [r for r in rows if r["config"] == name]
    f1s = [r["f1_macro"] for r in cfg_rows]
    resumen.append({"config": name, "modelo": model_name, "head": head,
                    "f1_macro_mean": round(np.mean(f1s), 4), "f1_macro_std": round(np.std(f1s), 4),
                    "f1_macro_ensemble": round(ens_f1, 4), "accuracy_ensemble": round(ens_acc, 4),
                    "f1_neutro_ensemble": round(ens_pc["neutro"], 4),
                    "f1_negativo_ensemble": round(ens_pc["negativo"], 4),
                    "f1_positivo_ensemble": round(ens_pc["positivo"], 4)})
    print(f"  >> ENSEMBLE f1_macro={ens_f1:.3f} acc={ens_acc:.3f} f1_neutro={ens_pc['neutro']:.3f}")
    pd.DataFrame(resumen).to_csv(REPORTS_DIR / "mejoras_ablacion_rapido_resumen.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
res = pd.DataFrame(resumen).sort_values("f1_macro_ensemble", ascending=False)
print("\n==================== RESUMEN ====================")
print(res.to_string(index=False))
if len(res):
    best = res.iloc[0]
    print(f"\nMEJOR: {best['config']}  ensemble F1-macro={best['f1_macro_ensemble']:.3f}  "
          f"(neutro={best['f1_neutro_ensemble']:.3f})")
    print(f"Referencia notebook actual (mBERT+CNN, gold antiguo): ~0.61")

# Grafico
if len(res):
    plt.figure(figsize=(10, 6))
    x = np.arange(len(res)); w = 0.38
    plt.bar(x - w/2, res["f1_macro_mean"], w, yerr=res["f1_macro_std"], capsize=4, label="Single (media±std, 3 semillas)")
    plt.bar(x + w/2, res["f1_macro_ensemble"], w, label="Ensemble (3 semillas)")
    plt.axhline(0.80, ls="--", color="green", alpha=0.7, label="Objetivo 0.80")
    plt.axhline(0.61, ls=":", color="gray", alpha=0.7, label="Notebook actual ~0.61")
    plt.xticks(x, res["config"], rotation=20, ha="right")
    plt.ylabel("F1-macro (test)"); plt.ylim(0, 1)
    plt.title("Ablacion de mejoras - F1-macro en test (404 ejemplos)")
    plt.legend(fontsize=8); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(VIS_DIR / "mejoras_ablacion_rapido_f1.png", dpi=180)
    print("\nGuardado:", VIS_DIR / "mejoras_ablacion_rapido_f1.png")
