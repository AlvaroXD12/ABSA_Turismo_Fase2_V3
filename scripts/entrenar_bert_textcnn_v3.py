# -*- coding: utf-8 -*-
# ============================================================
# Paso 7 - Entrenamiento completo BERT multilingual + TextCNN sobre gold v3,
# ENFOCADO a mejorar la clase negativa. (PREPARADO; lanzar tras confirmacion.)
# ------------------------------------------------------------
# - Arquitectura base oficial del PPI: mBERT + TextCNN (la ablacion mostro que el
#   encoder no es la palanca; la senal negativa si).
# - Foco en negativa: FocalLoss(gamma) + class weights inversos con BOOST extra a
#   la clase negativa (NEG_BOOST).
# - >=3 semillas; se reporta media +/- desv. de F1-macro, F1/recall negativo,
#   F1 neutro y F1 por aspecto. Ensemble por promedio de probabilidades.
# - Seleccion del mejor epoch por F1-macro en validation (metrica principal R1).
# - Evalua contra los umbrales de la spec (R1/R2) e imprime PASA/FALLA.
# Salidas (outputs/reports/ y outputs/visualizations/, sufijo _v3):
#   resultados_bert_textcnn_v3.csv (detalle por semilla)
#   resumen_bert_textcnn_v3.csv    (media+/-std + ensemble + veredicto)
#   classification_report_v3.csv, por_aspecto_v3.csv, matriz_confusion_v3.png
# ============================================================
from pathlib import Path
import random, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import (precision_recall_fscore_support, accuracy_score,
                             confusion_matrix, classification_report)

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"; REP = BASE / "outputs" / "reports"; VIS = BASE / "outputs" / "visualizations"
for d in (REP, VIS): d.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
L = ["negativo", "neutro", "positivo"]; L2I = {l: i for i, l in enumerate(L)}; I2L = {i: l for l, i in L2I.items()}

# ---- Config (knobs principales) ----
MODEL_NAME = "bert-base-multilingual-cased"
MAX_LEN, BATCH, EPOCHS = 256, 4, 12   # batch 4: fine-tuning completo de mBERT cabe en 4GB (+AMP)
LR, WD, WARMUP, PATIENCE, DROPOUT = 2e-5, 0.10, 0.10, 3, 0.40
CNN_FILTERS, CNN_KERNELS = 128, (2, 3, 4)
FOCAL_GAMMA = 2.0          # enfoca ejemplos dificiles (negativos confundidos)
NEG_BOOST = 1.5            # peso extra a la clase negativa por encima del inverso de frecuencia
SEEDS = [42, 7, 123]       # >=3 semillas
# Umbrales de la spec (modulo-absa-fase2.md)
TH_MACRO, TH_NEG_F1, TH_NEG_REC, TH_NEU_F1 = 0.70, 0.60, 0.60, 0.60


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def load(split):
    d = pd.read_csv(DATA / f"{split}_gold_v3.csv", encoding="utf-8-sig")
    d["label"] = d["label"].astype(str).str.lower().str.strip()
    if "input_modelo" not in d.columns or d["input_modelo"].isna().any():
        d["input_modelo"] = "aspecto: " + d["aspecto"].astype(str) + " reseña: " + d["text_clean"].astype(str)
    return d[d["label"].isin(L)].reset_index(drop=True)


train_df, val_df, test_df = load("train"), load("val"), load("test")
print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)} | device={DEVICE}")


class DS(Dataset):
    def __init__(self, t, l, tok): self.t=list(t); self.l=list(l); self.tok=tok
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
        # Gradient checkpointing: reduce drasticamente la memoria de activaciones
        # (necesario para fine-tuning completo de mBERT en 4GB). Cambia computo por memoria.
        self.bert.config.use_cache = False
        self.bert.gradient_checkpointing_enable()
        h = self.bert.config.hidden_size
        self.convs = nn.ModuleList([nn.Conv1d(h, CNN_FILTERS, k) for k in CNN_KERNELS])
        self.drop = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(CNN_FILTERS * len(CNN_KERNELS), len(L))
    def forward(self, ids, mask):
        x = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state.transpose(1, 2)
        pooled = [torch.max(torch.relu(c(x)), dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(pooled, dim=1)))


class FocalLoss(nn.Module):
    def __init__(self, weight, gamma): super().__init__(); self.w=weight; self.g=gamma
    def forward(self, logits, y):
        ce = nn.functional.cross_entropy(logits, y, weight=self.w, reduction="none")
        return (((1 - torch.exp(-ce)) ** self.g) * ce).mean()


def class_weights(labels):
    c = pd.Series(labels).value_counts().reindex(L, fill_value=0); tot = c.sum()
    w = [tot / (len(L) * c[l]) if c[l] > 0 else 0.0 for l in L]
    w[L2I["negativo"]] *= NEG_BOOST     # refuerzo extra a la negativa
    return torch.tensor(w, dtype=torch.float).to(DEVICE)


def metrics(trues, preds):
    pr, rc, f1, _ = precision_recall_fscore_support(trues, preds, labels=L, average=None, zero_division=0)
    _, _, mf1, _ = precision_recall_fscore_support(trues, preds, labels=L, average="macro", zero_division=0)
    out = {"f1_macro": mf1, "accuracy": accuracy_score(trues, preds)}
    for i, l in enumerate(L):
        out[f"f1_{l}"] = f1[i]; out[f"recall_{l}"] = rc[i]
    return out


def predict(model, loader):
    model.eval(); P, T = [], []
    with torch.no_grad():
        for b in loader:
            with torch.autocast("cuda", enabled=USE_AMP):
                lo = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE))
            P.append(torch.softmax(lo.float(), 1).cpu().numpy()); T += [I2L[y] for y in b["labels"].numpy()]
    return np.concatenate(P), T


def train_one(seed):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tl = DataLoader(DS(train_df["input_modelo"], train_df["label"], tok), batch_size=BATCH, shuffle=True)
    vl = DataLoader(DS(val_df["input_modelo"], val_df["label"], tok), batch_size=BATCH)
    el = DataLoader(DS(test_df["input_modelo"], test_df["label"], tok), batch_size=BATCH)
    model = BERTTextCNN().to(DEVICE)
    loss_fn = FocalLoss(class_weights(train_df["label"]), FOCAL_GAMMA)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    tot = len(tl) * EPOCHS
    sch = get_linear_schedule_with_warmup(opt, int(tot * WARMUP), tot)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    best, best_state, pat = -1, None, 0
    for ep in range(EPOCHS):
        model.train()
        for b in tl:
            opt.zero_grad()
            with torch.autocast("cuda", enabled=USE_AMP):
                loss = loss_fn(model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE)), b["labels"].to(DEVICE))
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
        vp, vt = predict(model, vl)
        vf = metrics(vt, [I2L[i] for i in vp.argmax(1)])["f1_macro"]
        if vf > best: best, best_state, pat = vf, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= PATIENCE: break
    if best_state: model.load_state_dict(best_state)
    tp, tt = predict(model, el); del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return tp, tt


def por_aspecto(test_df, preds):
    df = test_df.copy(); df["pred"] = preds; rows = []
    for asp, g in df.groupby("aspecto"):
        _, _, f1, _ = precision_recall_fscore_support(g["label"], g["pred"], labels=L, average="macro", zero_division=0)
        rows.append({"aspecto": asp, "soporte": len(g), "f1_macro": round(f1, 4)})
    return pd.DataFrame(rows).sort_values("f1_macro")


def main():
    rows, probs, tt = [], [], None
    for seed in SEEDS:
        p, t = train_one(seed); tt = t; probs.append(p)
        m = metrics(t, [I2L[i] for i in p.argmax(1)])
        rows.append({"seed": seed, **{k: round(v, 4) for k, v in m.items()}})
        print(f"  seed={seed} f1_macro={m['f1_macro']:.3f} f1_neg={m['f1_negativo']:.3f} "
              f"rec_neg={m['recall_negativo']:.3f} f1_neu={m['f1_neutro']:.3f}")
        pd.DataFrame(rows).to_csv(REP / "resultados_bert_textcnn_v3.csv", index=False, encoding="utf-8-sig")
    det = pd.DataFrame(rows)
    ens = np.mean(probs, 0); ep = [I2L[i] for i in ens.argmax(1)]; em = metrics(tt, ep)
    # resumen media+-std + ensemble + veredicto vs spec
    agg = {f"{c}_mean": round(det[c].mean(), 4) for c in det.columns if c != "seed"}
    agg.update({f"{c}_std": round(det[c].std(), 4) for c in det.columns if c != "seed"})
    veredicto = {
        "macro>=0.70": bool(em["f1_macro"] >= TH_MACRO),
        "neg_f1>=0.60": bool(em["f1_negativo"] >= TH_NEG_F1),
        "neg_recall>=0.60": bool(em["recall_negativo"] >= TH_NEG_REC),
        "neu_f1>=0.60": bool(em["f1_neutro"] >= TH_NEU_F1),
        "estable_std<=0.03": bool(det["f1_macro"].std() <= 0.03),
    }
    res = {**agg, **{f"ensemble_{k}": round(v, 4) for k, v in em.items()}, **{f"VEREDICTO_{k}": v for k, v in veredicto.items()}}
    pd.DataFrame([res]).to_csv(REP / "resumen_bert_textcnn_v3.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(classification_report(tt, ep, labels=L, output_dict=True, zero_division=0)).T.to_csv(
        REP / "classification_report_v3.csv", encoding="utf-8-sig")
    por_aspecto(test_df, ep).to_csv(REP / "por_aspecto_v3.csv", index=False, encoding="utf-8-sig")
    cm = confusion_matrix(tt, ep, labels=L)
    plt.figure(figsize=(5, 4)); plt.imshow(cm, cmap="Blues")
    plt.xticks(range(3), L, rotation=45); plt.yticks(range(3), L); plt.colorbar()
    for i in range(3):
        for j in range(3): plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.title("Matriz de confusión (ensemble) - gold v3"); plt.ylabel("real"); plt.xlabel("pred")
    plt.tight_layout(); plt.savefig(VIS / "matriz_confusion_v3.png", dpi=180)
    print("\n=== ENSEMBLE ===")
    print(f"  f1_macro={em['f1_macro']:.3f} f1_neg={em['f1_negativo']:.3f} rec_neg={em['recall_negativo']:.3f} f1_neu={em['f1_neutro']:.3f}")
    print("  VEREDICTO vs spec:", veredicto)
    print("Reportes en outputs/reports/*_v3.csv")


if __name__ == "__main__":
    main()
