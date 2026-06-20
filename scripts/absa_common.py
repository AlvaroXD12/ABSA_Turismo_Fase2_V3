# -*- coding: utf-8 -*-
# Núcleo compartido del módulo ABSA v4: lo usan TANTO el script de entrenamiento
# (entrenar_modelo_v4.py, un proceso por modelo) COMO el notebook (reporte visual).
# Así no hay código duplicado y cada modelo entrena en su propio proceso (GPU fresca).
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import random, time, gc, json
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# -------- Rutas --------
BASE = Path(__file__).resolve().parent.parent
DATA, REP, VIS = BASE / "data", BASE / "outputs" / "reports", BASE / "outputs" / "visualizations"
MODELS_DIR, PRED_DIR, MATR_DIR = BASE / "models", BASE / "outputs" / "predictions", BASE / "outputs" / "matrices"
for d in (REP, VIS, MODELS_DIR, PRED_DIR, MATR_DIR): d.mkdir(parents=True, exist_ok=True)

# -------- Config (consistente entre script y notebook) --------
VER = "v4"
MODELOS = {"xlmr": "xlm-roberta-base", "bert": "bert-base-multilingual-cased"}
MAX_LEN, BATCH, EPOCHS = 192, 4, 10
LR, WEIGHT_DECAY, WARMUP_RATIO, PATIENCE, DROPOUT = 2e-5, 0.10, 0.10, 3, 0.40
CNN_FILTERS, CNN_KERNELS = 128, (2, 3, 4)
SEEDS = [42, 7, 123, 2024, 77]
LABEL_SMOOTHING = 0.1
USE_GRADIENT_CHECKPOINTING = True
CALIBRAR_DECISION = True
NEG_BOOST_GRID, FOCAL_GRID, SEARCH_EPOCHS = [1.2, 1.8], [2.0], 3
TH_MACRO, TH_NEG_F1, TH_NEG_REC, TH_NEU_F1 = 0.70, 0.60, 0.60, 0.60

LABELS = ["negativo", "neutro", "positivo"]; L2I = {l: i for i, l in enumerate(LABELS)}; I2L = {i: l for l, i in L2I.items()}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu"); USE_AMP = torch.cuda.is_available()

def art(tag, kind):
    return {"det": REP / f"resultados_{tag}_{VER}.csv", "resumen": REP / f"resumen_{tag}_{VER}.csv",
            "aspecto": REP / f"por_aspecto_{tag}_{VER}.csv", "preds": PRED_DIR / f"predicciones_test_{tag}_{VER}.csv",
            "hist": REP / f"historial_{tag}_{VER}.csv", "bias": MODELS_DIR / f"_bias_{tag}_{VER}.npy"}[kind]
HP_FILE = REP / f"hp_seleccion_{VER}.csv"

# -------- Datos --------
def load_splits():
    out = {}
    for nm in ("train", "val", "test"):
        d = pd.read_csv(DATA / f"{nm}_gold_{VER}.csv", encoding="utf-8-sig")
        d["label"] = d["label"].astype(str).str.lower().str.strip()
        if "input_modelo" not in d.columns or d["input_modelo"].isna().any():
            d["input_modelo"] = "aspecto: " + d["aspecto"].astype(str) + " reseña: " + d["text_clean"].astype(str)
        out[nm] = d
    return out["train"], out["val"], out["test"]

# -------- Modelo / loss / métricas --------
class ABSADataset(Dataset):
    def __init__(s, texts, labels, tok): s.t=list(texts); s.l=list(labels) if labels is not None else None; s.tok=tok
    def __len__(s): return len(s.t)
    def __getitem__(s, i):
        e = s.tok(str(s.t[i]), add_special_tokens=True, max_length=MAX_LEN, padding="max_length", truncation=True, return_attention_mask=True, return_tensors="pt")
        it = {"input_ids": e["input_ids"].squeeze(0), "attention_mask": e["attention_mask"].squeeze(0)}
        if s.l is not None: it["labels"] = torch.tensor(L2I[s.l[i]], dtype=torch.long)
        return it

class TextCNN(nn.Module):
    def __init__(s, model_name):
        super().__init__(); s.bert = AutoModel.from_pretrained(model_name)
        if USE_GRADIENT_CHECKPOINTING: s.bert.config.use_cache=False; s.bert.gradient_checkpointing_enable()
        h = s.bert.config.hidden_size
        s.convs = nn.ModuleList([nn.Conv1d(h, CNN_FILTERS, k) for k in CNN_KERNELS])
        s.drop = nn.Dropout(DROPOUT); s.fc = nn.Linear(CNN_FILTERS*len(CNN_KERNELS), 3)
    def forward(s, ids, mask):
        x = s.bert(input_ids=ids, attention_mask=mask).last_hidden_state.transpose(1, 2)
        return s.fc(s.drop(torch.cat([torch.max(torch.relu(c(x)), 2).values for c in s.convs], 1)))

class FocalLoss(nn.Module):
    def __init__(s, weight, gamma): super().__init__(); s.w=weight; s.g=gamma
    def forward(s, logits, y):
        ce = nn.functional.cross_entropy(logits, y, weight=s.w, reduction="none", label_smoothing=LABEL_SMOOTHING)
        return (((1-torch.exp(-ce))**s.g)*ce).mean()

def class_weights(labels, neg_boost):
    c = pd.Series(labels).value_counts().reindex(LABELS, fill_value=0); tot=c.sum()
    w = [tot/(3*c[l]) if c[l]>0 else 0.0 for l in LABELS]; w[L2I["negativo"]] *= neg_boost
    return torch.tensor(w, dtype=torch.float).to(DEVICE)

def metrics(trues, preds):
    pr, rc, f1, _ = precision_recall_fscore_support(trues, preds, labels=LABELS, average=None, zero_division=0)
    _, _, mf1, _ = precision_recall_fscore_support(trues, preds, labels=LABELS, average="macro", zero_division=0)
    o = {"f1_macro": mf1, "accuracy": accuracy_score(trues, preds)}
    for i, l in enumerate(LABELS): o[f"f1_{l}"]=f1[i]; o[f"recall_{l}"]=rc[i]
    return o

def set_seed(sd):
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(sd)

def predict(model, loader, loss_fn=None):
    model.eval(); P, T, tot = [], [], 0.0
    with torch.no_grad():
        for b in loader:
            ids, mask = b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE)
            with torch.autocast("cuda", enabled=USE_AMP):
                lo = model(ids, mask)
                if loss_fn is not None and "labels" in b: tot += loss_fn(lo, b["labels"].to(DEVICE)).item()
            P.append(torch.softmax(lo.float(),1).cpu().numpy())
            if "labels" in b: T += [I2L[i] for i in b["labels"].numpy()]
    return np.concatenate(P), T, tot/max(len(loader),1)

def apply_bias(probs, bias): return [I2L[i] for i in (np.log(probs+1e-9)+bias).argmax(1)]
def best_bias(vp, vt):
    if not CALIBRAR_DECISION: return np.zeros(3)
    logp=np.log(vp+1e-9); g=np.arange(-1.2,1.21,0.2); best,bb=-1,np.zeros(3)
    for b0 in g:
        for b1 in g:
            b=np.array([b0,b1,0.0]); f=metrics(vt,[I2L[i] for i in (logp+b).argmax(1)])["f1_macro"]
            if f>best: best,bb=f,b
    return bb

def train_one(seed, model_name, neg_boost, focal_gamma, train, val, test, epochs=EPOCHS, record_history=False, save_tag=None):
    set_seed(seed); tok = AutoTokenizer.from_pretrained(model_name); short = model_name.split("-")[0]
    tl = DataLoader(ABSADataset(train["input_modelo"], train["label"], tok), batch_size=BATCH, shuffle=True)
    vl = DataLoader(ABSADataset(val["input_modelo"], val["label"], tok), batch_size=BATCH)
    el = DataLoader(ABSADataset(test["input_modelo"], test["label"], tok), batch_size=BATCH)
    model = TextCNN(model_name).to(DEVICE); loss_fn = FocalLoss(class_weights(train["label"], neg_boost), focal_gamma)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = get_linear_schedule_with_warmup(opt, int(len(tl)*epochs*WARMUP_RATIO), len(tl)*epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP); best,bs,pat,hist = -1,None,0,[]
    for ep in range(1, epochs+1):
        model.train(); run=0.0; nbt=len(tl); cada=max(1, nbt//4)
        for bi, b in enumerate(tl, 1):
            opt.zero_grad()
            with torch.autocast("cuda", enabled=USE_AMP):
                loss = loss_fn(model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE)), b["labels"].to(DEVICE))
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update(); sch.step(); run += loss.item()
            if bi % cada == 0 or bi == nbt:
                print(f"      {short} seed {seed} ép {ep}/{epochs}: batch {bi}/{nbt} ({bi*100//nbt}%) loss={loss.item():.3f}", flush=True)
        vp, vt, vloss = predict(model, vl, loss_fn); vf = metrics(vt, [I2L[i] for i in vp.argmax(1)])["f1_macro"]
        print(f"  [{short}] seed {seed} | época {ep}/{epochs} | val_f1={vf:.3f} | loss={run/len(tl):.3f}", flush=True)
        if record_history:
            tp_, tt_, _ = predict(model, tl); tf = metrics(tt_, [I2L[i] for i in tp_.argmax(1)])["f1_macro"]
            hist.append({"epoch": ep, "train_loss": run/len(tl), "val_loss": vloss, "train_f1_macro": tf, "val_f1_macro": vf})
        if vf > best: best,bs,pat = vf, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= PATIENCE: break
    if bs: model.load_state_dict(bs)
    if save_tag: torch.save(bs, MODELS_DIR / f"modelo_{save_tag}.pt")
    tp, tt, _ = predict(model, el); vp, vtv, _ = predict(model, vl); del model, opt, scaler; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return tp, tt, vp, vtv, pd.DataFrame(hist)

def get_hp(train, val, test):
    # HP siempre seleccionado en el candidato principal (XLM-R), por validacion. Cacheado en HP_FILE.
    if HP_FILE.exists():
        r = pd.read_csv(HP_FILE).iloc[0]; return float(r["neg_boost"]), float(r["focal_gamma"])
    combos = [(nb, fg) for nb in NEG_BOOST_GRID for fg in FOCAL_GRID]; rows = []
    print(f"[Búsqueda HP] {len(combos)} combos en validación (sobre XLM-R)...", flush=True)
    for ci, (nb, fg) in enumerate(combos, 1):
        print(f"  combo {ci}/{len(combos)}: NEG_BOOST={nb} FOCAL={fg}", flush=True)
        _, _, vp, vtv, _ = train_one(42, MODELOS["xlmr"], nb, fg, train, val, test, epochs=SEARCH_EPOCHS)
        vf = metrics(vtv, [I2L[i] for i in vp.argmax(1)])["f1_macro"]
        rows.append({"neg_boost": nb, "focal_gamma": fg, "val_f1_macro": round(vf, 4)})
        print(f"    -> val_f1={vf:.3f}", flush=True)
    hp = pd.DataFrame(rows).sort_values("val_f1_macro", ascending=False); hp.to_csv(HP_FILE, index=False, encoding="utf-8-sig")
    return float(hp.iloc[0]["neg_boost"]), float(hp.iloc[0]["focal_gamma"])

def run_modelo(tag, train, val, test, neg_boost, focal_gamma):
    from sklearn.metrics import classification_report
    model_name = MODELOS[tag]
    print(f"\n===== Entrenando {tag.upper()} ({model_name}) — {len(SEEDS)} semillas | NEG_BOOST={neg_boost} FOCAL={focal_gamma} =====", flush=True)
    rows, probs, vprobs, tt0, vt0, hist0 = [], [], [], None, None, None
    for k, sd in enumerate(SEEDS):
        print(f"  [{tag}] semilla {sd} ({k+1}/{len(SEEDS)})", flush=True)
        tp, tt, vp, vtv, h = train_one(sd, model_name, neg_boost, focal_gamma, train, val, test, record_history=(k==0), save_tag=f"{tag}_seed{sd}_{VER}")
        tt0, vt0 = tt, vtv; probs.append(tp); vprobs.append(vp)
        if k == 0: hist0 = h
        m = metrics(tt, [I2L[i] for i in tp.argmax(1)]); rows.append({"seed": sd, **{kk:round(v,4) for kk,v in m.items()}})
    det = pd.DataFrame(rows); ens=np.mean(probs,0); ensv=np.mean(vprobs,0)
    bias = best_bias(ensv, vt0); preds = apply_bias(ens, bias); em = metrics(tt0, preds)
    det.to_csv(art(tag,"det"), index=False, encoding="utf-8-sig"); hist0.to_csv(art(tag,"hist"), index=False, encoding="utf-8-sig")
    np.save(art(tag,"bias"), bias); pd.DataFrame({"y_true": tt0, "y_pred": preds}).to_csv(art(tag,"preds"), index=False, encoding="utf-8-sig")
    cols=[c for c in det.columns if c!="seed"]
    pd.DataFrame([{"modelo": tag, **{f"ensemble_{k}":round(v,4) for k,v in em.items()},
                   "media_f1_macro": round(det["f1_macro"].mean(),4), "std_f1_macro": round(det["f1_macro"].std(),4),
                   "estable_std<=0.03": bool(det["f1_macro"].std()<=0.03)}]).to_csv(art(tag,"resumen"), index=False, encoding="utf-8-sig")
    ta = test.copy(); ta["pred"] = preds; fa=[]
    for asp, gg in ta.groupby("aspecto"):
        _,_,f1,_ = precision_recall_fscore_support(gg["label"], gg["pred"], labels=LABELS, average="macro", zero_division=0)
        fa.append({"aspecto":asp, "soporte":len(gg), "f1_macro":round(f1,4)})
    pd.DataFrame(fa).sort_values("f1_macro").to_csv(art(tag,"aspecto"), index=False, encoding="utf-8-sig")
    print(f"  [{tag}] LISTO. ensemble F1-macro={em['f1_macro']:.4f} (std {det['f1_macro'].std():.4f})", flush=True)
    return em
