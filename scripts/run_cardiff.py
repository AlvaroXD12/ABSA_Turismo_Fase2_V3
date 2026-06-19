# -*- coding: utf-8 -*-
# Corre SOLO la configuracion xlmr_sent_simple (cardiffnlp twitter-xlm-roberta
# preentrenado en sentimiento), que fallo en la ablacion rapida por el tokenizer.
# Workaround: tokenizer de xlm-roberta-base (mismo vocab) + pesos en safetensors.
# Mismos hiperparametros que mejoras_ablacion_rapido.py para comparabilidad.
from pathlib import Path
import random, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"; REPORTS_DIR = BASE_DIR / "outputs" / "reports"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
VALID_LABELS = ["negativo", "neutro", "positivo"]
L2I = {l: i for i, l in enumerate(VALID_LABELS)}; I2L = {i: l for l, i in L2I.items()}

MAX_LEN, BATCH, EPOCHS, LR, WD, WARMUP, PATIENCE, DROPOUT, FREEZE = 160, 16, 4, 2.5e-5, 0.10, 0.10, 2, 0.40, 6
SEEDS = [42, 7]
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
TOKENIZER_NAME = "xlm-roberta-base"   # mismo vocab que el modelo cardiff

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def load_split(name):
    df = pd.read_csv(DATA_DIR / f"{name}.csv", encoding="utf-8-sig")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    if "input_modelo" not in df.columns or df["input_modelo"].isna().any():
        df["input_modelo"] = "aspecto: " + df["aspecto"].astype(str) + " reseña: " + df["text_clean"].astype(str)
    return df[df["label"].isin(VALID_LABELS)].reset_index(drop=True)

train_df, val_df, test_df = load_split("train"), load_split("validation"), load_split("test")
print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

class DS(Dataset):
    def __init__(self, t, l, tok): self.t=list(t); self.l=list(l); self.tok=tok
    def __len__(self): return len(self.t)
    def __getitem__(self, i):
        e=self.tok(str(self.t[i]), add_special_tokens=True, max_length=MAX_LEN, padding="max_length",
                   truncation=True, return_attention_mask=True, return_tensors="pt")
        return {"input_ids": e["input_ids"].squeeze(0), "attention_mask": e["attention_mask"].squeeze(0),
                "labels": torch.tensor(L2I[self.l[i]], dtype=torch.long)}

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL_NAME, use_safetensors=True)
        if hasattr(self.enc, "embeddings"):
            for p in self.enc.embeddings.parameters(): p.requires_grad = False
        for layer in self.enc.encoder.layer[:FREEZE]:
            for p in layer.parameters(): p.requires_grad = False
        self.drop = nn.Dropout(DROPOUT)
        self.cls = nn.Linear(self.enc.config.hidden_size, len(VALID_LABELS))
    def forward(self, ids, mask):
        x = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.cls(self.drop(x))

def cw(labels):
    c = pd.Series(labels).value_counts().reindex(VALID_LABELS, fill_value=0); tot = c.sum()
    return torch.tensor([tot/(3*c[l]) if c[l]>0 else 0.0 for l in VALID_LABELS], dtype=torch.float).to(DEVICE)

def predict(model, loader):
    model.eval(); P, T = [], []
    with torch.no_grad():
        for b in loader:
            with torch.autocast("cuda", enabled=USE_AMP):
                lo = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE))
            P.append(torch.softmax(lo.float(), 1).cpu().numpy()); T += [I2L[y] for y in b["labels"].numpy()]
    return np.concatenate(P), T

def f1m(t, p):
    return precision_recall_fscore_support(t, p, labels=VALID_LABELS, average="macro", zero_division=0)[2]
def pcf1(t, p):
    f = precision_recall_fscore_support(t, p, labels=VALID_LABELS, average=None, zero_division=0)[2]
    return dict(zip(VALID_LABELS, f))

def train_one(seed):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tl = DataLoader(DS(train_df["input_modelo"], train_df["label"], tok), batch_size=BATCH, shuffle=True)
    vl = DataLoader(DS(val_df["input_modelo"], val_df["label"], tok), batch_size=BATCH)
    el = DataLoader(DS(test_df["input_modelo"], test_df["label"], tok), batch_size=BATCH)
    model = Model().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=cw(train_df["label"]))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    tot = len(tl)*EPOCHS
    sch = get_linear_schedule_with_warmup(opt, int(tot*WARMUP), tot)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    best, best_state, pat = -1, None, 0
    for ep in range(EPOCHS):
        model.train()
        for b in tl:
            opt.zero_grad()
            with torch.autocast("cuda", enabled=USE_AMP):
                loss = loss_fn(model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE)), b["labels"].to(DEVICE))
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); sch.step()
        vp, vt = predict(model, vl); vf = f1m(vt, [I2L[i] for i in vp.argmax(1)])
        if vf > best: best, best_state, pat = vf, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= PATIENCE: break
    if best_state: model.load_state_dict(best_state)
    tp, tt = predict(model, el); del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return tp, tt

rows, probs = [], []
tt = None
for seed in SEEDS:
    p, t = train_one(seed); tt = t; probs.append(p)
    pr = [I2L[i] for i in p.argmax(1)]; pc = pcf1(t, pr)
    rows.append({"config": "xlmr_sent_simple", "modelo": MODEL_NAME, "head": "simple", "seed": seed,
                 "f1_macro": round(f1m(t, pr), 4), "accuracy": round(accuracy_score(t, pr), 4),
                 "f1_negativo": round(pc["negativo"], 4), "f1_neutro": round(pc["neutro"], 4),
                 "f1_positivo": round(pc["positivo"], 4)})
    print(f"  seed={seed} f1_macro={f1m(t,pr):.3f} acc={accuracy_score(t,pr):.3f} f1_neutro={pc['neutro']:.3f}")

ens = np.mean(probs, 0); ep = [I2L[i] for i in ens.argmax(1)]; epc = pcf1(tt, ep)
print(f"  >> ENSEMBLE f1_macro={f1m(tt,ep):.3f} acc={accuracy_score(tt,ep):.3f} f1_neutro={epc['neutro']:.3f} f1_neg={epc['negativo']:.3f}")
pd.DataFrame(rows).to_csv(REPORTS_DIR / "cardiff_resultado.csv", index=False, encoding="utf-8-sig")
print("guardado:", REPORTS_DIR / "cardiff_resultado.csv")
