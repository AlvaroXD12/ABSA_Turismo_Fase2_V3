# -*- coding: utf-8 -*-
# ============================================================
# Curva de aprendizaje del modelo BERT multilingual + TextCNN
# ------------------------------------------------------------
# Objetivo: justificar empiricamente cuanto mas hay que anotar.
# Entrena el MISMO modelo del notebook 02 con fracciones crecientes
# del train (manteniendo validation y test FIJOS y completos) y mide
# F1-macro en validation y test. Repite cada punto con varias semillas
# para reportar media +/- desviacion (imprescindible con n pequeno).
#
# Salidas:
#   outputs/reports/curva_aprendizaje_detalle.csv   (una fila por fraccion x semilla)
#   outputs/reports/curva_aprendizaje_resumen.csv   (media/std por fraccion)
#   outputs/visualizations/curva_aprendizaje_bert_textcnn.png
#
# Arquitectura, loss, optimizador y early stopping son identicos al
# notebook 02 para que la curva sea comparable con el modelo final.
# ============================================================

from pathlib import Path
import random
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

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ------------------------------------------------------------
# Configuracion (igual que notebook 02)
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"
VIS_DIR = BASE_DIR / "outputs" / "visualizations"
for d in [REPORTS_DIR, VIS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "bert-base-multilingual-cased"
MAX_LEN = 256
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.10
WARMUP_RATIO = 0.10
PATIENCE = 2
FOCAL_GAMMA = 0.0
CNN_NUM_FILTERS = 128
CNN_KERNEL_SIZES = (2, 3, 4)
DROPOUT = 0.45

VALID_LABELS = ["negativo", "neutro", "positivo"]
LABEL_TO_ID = {l: i for i, l in enumerate(VALID_LABELS)}
ID_TO_LABEL = {i: l for l, i in LABEL_TO_ID.items()}

# Parametros del experimento de curva
FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
SEEDS = [42, 7, 123]                 # 3 semillas por punto -> barras de error
PLATEAU_DELTA_F1 = 0.01              # mejora marginal < 1% F1 => meseta

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Datos
# ------------------------------------------------------------
def load_split(name):
    df = pd.read_csv(DATA_DIR / f"{name}.csv", encoding="utf-8-sig")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    if "input_modelo" not in df.columns or df["input_modelo"].isna().any():
        df["input_modelo"] = "aspecto: " + df["aspecto"].astype(str) + " reseña: " + df["text_clean"].astype(str)
    return df[df["label"].isin(VALID_LABELS)].reset_index(drop=True)


train_full = load_split("train")
validation_df = load_split("validation")
test_df = load_split("test")
print(f"train={len(train_full)}  validation={len(validation_df)}  test={len(test_df)}")


def stratified_subsample(df, frac, seed):
    """Submuestra estratificada por label conservando proporciones."""
    if frac >= 1.0:
        return df.reset_index(drop=True)
    parts = []
    for label, g in df.groupby("label"):
        n = max(1, int(round(len(g) * frac)))
        parts.append(g.sample(n=min(n, len(g)), random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ------------------------------------------------------------
# Dataset / Modelo / Loss (identicos al nb02)
# ------------------------------------------------------------
class ABSADataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]), add_special_tokens=True, max_length=self.max_len,
            padding="max_length", truncation=True, return_attention_mask=True, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(LABEL_TO_ID[self.labels[idx]], dtype=torch.long),
        }


class BERTTextCNN(nn.Module):
    def __init__(self, model_name, num_labels, num_filters, kernel_sizes, dropout):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.convs = nn.ModuleList([nn.Conv1d(hidden, num_filters, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [torch.max(torch.relu(conv(x)), dim=2).values for conv in self.convs]
        x = self.dropout(torch.cat(pooled, dim=1))
        return self.classifier(x)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def compute_class_weights(labels):
    counts = pd.Series(labels).value_counts().reindex(VALID_LABELS, fill_value=0)
    total = counts.sum()
    w = [total / (len(VALID_LABELS) * c) if c > 0 else 0.0 for c in counts]
    return torch.tensor(w, dtype=torch.float).to(DEVICE)


def f1_macro(y_true, y_pred):
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=VALID_LABELS, average="macro", zero_division=0)
    return f1


def run_epoch(model, loader, loss_fn, optimizer=None, scheduler=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    preds, trues = [], []
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        with torch.set_grad_enabled(train):
            logits = model(ids, mask)
            loss = loss_fn(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
        preds += [ID_TO_LABEL[p] for p in torch.argmax(logits, 1).cpu().numpy()]
        trues += [ID_TO_LABEL[y] for y in labels.cpu().numpy()]
    return f1_macro(trues, preds), trues, preds


def train_and_eval(train_df, seed):
    """Entrena con early stopping y devuelve (val_f1_best, test_f1, test_acc)."""
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tr_loader = DataLoader(ABSADataset(train_df["input_modelo"], train_df["label"], tokenizer, MAX_LEN),
                           batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(ABSADataset(validation_df["input_modelo"], validation_df["label"], tokenizer, MAX_LEN),
                           batch_size=BATCH_SIZE)
    te_loader = DataLoader(ABSADataset(test_df["input_modelo"], test_df["label"], tokenizer, MAX_LEN),
                           batch_size=BATCH_SIZE)

    model = BERTTextCNN(MODEL_NAME, len(VALID_LABELS), CNN_NUM_FILTERS, CNN_KERNEL_SIZES, DROPOUT).to(DEVICE)
    loss_fn = FocalLoss(weight=compute_class_weights(train_df["label"]), gamma=FOCAL_GAMMA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(tr_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)

    best_val, best_state, patience = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        run_epoch(model, tr_loader, loss_fn, optimizer, scheduler)
        val_f1, _, _ = run_epoch(model, va_loader, loss_fn)
        if val_f1 > best_val:
            best_val = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_f1, t_true, t_pred = run_epoch(model, te_loader, loss_fn)
    test_acc = accuracy_score(t_true, t_pred)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_val, test_f1, test_acc


# ------------------------------------------------------------
# Bucle de la curva
# ------------------------------------------------------------
rows = []
for frac in FRACTIONS:
    for seed in SEEDS:
        sub = stratified_subsample(train_full, frac, seed)
        val_f1, test_f1, test_acc = train_and_eval(sub, seed)
        rows.append({
            "fraccion": frac, "n_train": len(sub), "seed": seed,
            "val_f1_macro": round(val_f1, 4), "test_f1_macro": round(test_f1, 4),
            "test_accuracy": round(test_acc, 4),
        })
        print(f"  frac={frac:.1f} n={len(sub):3d} seed={seed:3d} "
              f"val_f1={val_f1:.3f} test_f1={test_f1:.3f} test_acc={test_acc:.3f}")
        # guardado incremental por si se interrumpe
        pd.DataFrame(rows).to_csv(REPORTS_DIR / "curva_aprendizaje_detalle.csv", index=False, encoding="utf-8-sig")

detalle = pd.DataFrame(rows)

# ------------------------------------------------------------
# Resumen (media +/- std por fraccion)
# ------------------------------------------------------------
resumen = detalle.groupby(["fraccion", "n_train"]).agg(
    val_f1_mean=("val_f1_macro", "mean"), val_f1_std=("val_f1_macro", "std"),
    test_f1_mean=("test_f1_macro", "mean"), test_f1_std=("test_f1_macro", "std"),
    test_acc_mean=("test_accuracy", "mean"), test_acc_std=("test_accuracy", "std"),
).reset_index().round(4)
resumen.to_csv(REPORTS_DIR / "curva_aprendizaje_resumen.csv", index=False, encoding="utf-8-sig")
print("\n=== RESUMEN ===")
print(resumen.to_string(index=False))

# Diagnostico de meseta: mejora marginal entre los dos ultimos puntos
if len(resumen) >= 2:
    last = resumen.iloc[-1]["test_f1_mean"]
    prev = resumen.iloc[-2]["test_f1_mean"]
    delta = last - prev
    if delta < PLATEAU_DELTA_F1:
        diag = (f"MESETA: la mejora del ultimo tramo es {delta:+.4f} F1 (< {PLATEAU_DELTA_F1}). "
                f"Anotar mas aporta poco con el esquema actual.")
    else:
        diag = (f"AUN EN PENDIENTE: el ultimo tramo sube {delta:+.4f} F1 (>= {PLATEAU_DELTA_F1}). "
                f"Anotar mas probablemente mejora el modelo.")
    print("\nDIAGNOSTICO:", diag)

# ------------------------------------------------------------
# Grafico con barras de error
# ------------------------------------------------------------
plt.figure(figsize=(9, 5.5))
plt.errorbar(resumen["n_train"], resumen["test_f1_mean"], yerr=resumen["test_f1_std"],
             marker="o", capsize=4, label="Test F1-macro")
plt.errorbar(resumen["n_train"], resumen["val_f1_mean"], yerr=resumen["val_f1_std"],
             marker="s", capsize=4, label="Validation F1-macro")
plt.title("Curva de aprendizaje - BERT multilingual + TextCNN\n(media +/- desv. de %d semillas)" % len(SEEDS))
plt.xlabel("Numero de ejemplos de entrenamiento anotados")
plt.ylabel("F1-macro")
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(VIS_DIR / "curva_aprendizaje_bert_textcnn.png", dpi=180)
print("\nGuardado:")
print(" ", REPORTS_DIR / "curva_aprendizaje_detalle.csv")
print(" ", REPORTS_DIR / "curva_aprendizaje_resumen.csv")
print(" ", VIS_DIR / "curva_aprendizaje_bert_textcnn.png")
