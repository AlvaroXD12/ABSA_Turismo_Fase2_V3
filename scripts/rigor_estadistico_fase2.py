# -*- coding: utf-8 -*-
# ============================================================
# Paquete de RIGOR ESTADÍSTICO (análisis posterior, NO tuning).
# No re-entrena, no cambia hiperparámetros, no toca el test. Solo INFERENCIA con
# los checkpoints v4 ya guardados + análisis estadístico.
#  1) Intervalos de confianza (bootstrap) del F1-macro y por clase (XLM-R).
#  2) Test de McNemar XLM-R vs BERT (errores pareados, mismo test).
#  3) Análisis cualitativo de errores (con texto, aspecto, confianza).
#  4) Gráficos + reporte.
# ============================================================
from pathlib import Path
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

import absa_common as ac
BASE = ac.BASE; OUT = BASE / "outputs" / "rigor_estadistico"; OUT.mkdir(parents=True, exist_ok=True)
L = ac.LABELS; RNG = np.random.default_rng(42)

def f1_macro(yt, yp): return precision_recall_fscore_support(yt, yp, labels=L, average="macro", zero_division=0)[2]
def per_class(yt, yp):
    pr, rc, f1, _ = precision_recall_fscore_support(yt, yp, labels=L, average=None, zero_division=0)
    return {f"f1_{l}": f1[i] for i, l in enumerate(L)} | {f"recall_{l}": rc[i] for i, l in enumerate(L)}

# ---------- Inferencia (ensemble de 5 semillas) con checkpoints guardados ----------
def infer_probs(tag, texts):
    mname = ac.MODELOS[tag]; tok = AutoTokenizer.from_pretrained(mname)
    loader = DataLoader(ac.ABSADataset(texts, None, tok), batch_size=16)
    acc = np.zeros((len(texts), 3)); n = 0
    for sd in ac.SEEDS:
        mp = ac.MODELS_DIR / f"modelo_{tag}_seed{sd}_{ac.VER}.pt"
        if not mp.exists(): continue
        model = ac.TextCNN(mname).to(ac.DEVICE); model.load_state_dict(torch.load(mp, map_location="cpu")); model.eval(); n += 1; pos = 0
        with torch.no_grad():
            for b in loader:
                with torch.autocast("cuda", enabled=ac.USE_AMP):
                    lo = model(b["input_ids"].to(ac.DEVICE), b["attention_mask"].to(ac.DEVICE))
                pr = torch.softmax(lo.float(), 1).cpu().numpy(); acc[pos:pos+len(pr)] += pr; pos += len(pr)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return acc / max(n, 1)

train, val, test = ac.load_splits()
yt = test["label"].tolist()
print("Infiriendo XLM-R y BERT (ensemble 5 semillas, calibración por validación)...", flush=True)
out = {}
for tag in ("xlmr", "bert"):
    vp = infer_probs(tag, val["input_modelo"].tolist()); tp = infer_probs(tag, test["input_modelo"].tolist())
    bias = ac.best_bias(vp, val["label"].tolist())
    preds = ac.apply_bias(tp, bias); conf = tp.max(1)
    out[tag] = {"probs": tp, "preds": preds, "conf": conf}
    pd.DataFrame({"y_true": yt, "y_pred": preds, "confianza": conf.round(4)}).to_csv(
        ac.PRED_DIR / f"predicciones_test_{tag}_{ac.VER}.csv", index=False, encoding="utf-8-sig")
    print(f"  {tag}: F1-macro={f1_macro(yt, preds):.4f}", flush=True)
yp_x, yp_b = out["xlmr"]["preds"], out["bert"]["preds"]

# ---------- 1) Bootstrap CIs (XLM-R) ----------
B = 2000; idx = np.arange(len(yt)); yt_a = np.array(yt); yp_a = np.array(yp_x)
boot = {"f1_macro": [], "f1_negativo": [], "recall_negativo": [], "f1_neutro": []}
for _ in range(B):
    s = RNG.choice(idx, size=len(idx), replace=True)
    pr, rc, f1, _ = precision_recall_fscore_support(yt_a[s], yp_a[s], labels=L, average=None, zero_division=0)
    boot["f1_macro"].append(f1.mean()); boot["f1_negativo"].append(f1[0]); boot["recall_negativo"].append(rc[0]); boot["f1_neutro"].append(f1[1])
ci_rows = []
for k, v in boot.items():
    v = np.array(v); ci_rows.append({"metrica": k, "puntual": round(np.mean(v), 4),
        "IC95_inf": round(np.percentile(v, 2.5), 4), "IC95_sup": round(np.percentile(v, 97.5), 4)})
ci = pd.DataFrame(ci_rows); ci.to_csv(OUT / "metricas_bootstrap.csv", index=False, encoding="utf-8-sig")
print("\n=== IC 95% bootstrap (XLM-R) ==="); print(ci.to_string(index=False))

# ---------- 2) McNemar ----------
cx = (yp_a == yt_a); cb = (np.array(yp_b) == yt_a)
b = int(np.sum(cx & ~cb)); c = int(np.sum(~cx & cb))   # b: XLM-R acierta y BERT no | c: BERT acierta y XLM-R no
n00 = int(np.sum(~cx & ~cb)); n11 = int(np.sum(cx & cb))
from scipy.stats import chi2, binomtest
stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
p_chi = float(chi2.sf(stat, 1)); p_exact = float(binomtest(min(b, c), b + c, 0.5).pvalue) if (b + c) > 0 else 1.0
mc = {"contingencia": {"ambos_correctos": n11, "XLMR_si_BERT_no": b, "BERT_si_XLMR_no": c, "ambos_incorrectos": n00},
      "n_discordantes": b + c, "mcnemar_chi2_cc": round(stat, 4), "p_value_chi2": round(p_chi, 6),
      "p_value_exacto_binomial": round(p_exact, 6),
      "interpretacion": ("XLM-R mejora a BERT de forma estadísticamente significativa (p<0.05)" if p_exact < 0.05 and b > c
                         else "no hay evidencia significativa de diferencia (p>=0.05)"),
      "nota": "Compara errores PAREADOS en el MISMO test v4 (610 instancias)."}
json.dump(mc, open(OUT / "mcnemar_xlmr_vs_bert.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n=== McNemar XLM-R vs BERT ==="); print(json.dumps(mc, ensure_ascii=False, indent=2))

# ---------- 3) Análisis cualitativo de errores (XLM-R) ----------
ASP_DEBIL = {"alojamiento", "clima", "aforo_multitudes", "seguridad", "gastronomia"}
NEG = ["no ", "mal", "caro", "sucio", "lleno", "frio", "calor", "lluvia", "pesimo", "horrible", "decepcion", "tarde", "espera", "cola", "dirty", "crowded", "bad", "worst", "rude"]
POS = ["excelente", "hermoso", "bonito", "increible", "recomiendo", "limpio", "tranquilo", "bueno", "genial", "amazing", "great", "beautiful", "clean", "nice", "best"]
def patron(texto, asp):
    t = str(texto).lower(); npos = sum(w in t for w in POS); nneg = sum(w in t for w in NEG)
    if asp in ASP_DEBIL: return "aspecto_debil"
    if npos >= 1 and nneg >= 1: return "resena_mixta"
    if len(t.split()) < 15: return "corta_falta_contexto"
    return "ambiguedad_semantica"
err = test.copy(); err["y_true"] = yt; err["y_pred"] = yp_x; err["confianza"] = out["xlmr"]["conf"].round(4)
err = err[err["y_true"] != err["y_pred"]].copy()
err["tipo_error"] = err["y_true"] + "→" + err["y_pred"]
err["patron_tentativo"] = err.apply(lambda r: patron(r["text_clean"], r["aspecto"]), axis=1)
cols = ["destination", "aspecto", "y_true", "y_pred", "tipo_error", "confianza", "patron_tentativo", "text_clean"]
err[cols].sort_values(["tipo_error", "confianza"], ascending=[True, False]).to_csv(OUT / "errores_cualitativos.csv", index=False, encoding="utf-8-sig")
print(f"\nErrores totales XLM-R: {len(err)} de {len(test)} | por tipo:", err["tipo_error"].value_counts().to_dict())
print("Patrones tentativos:", err["patron_tentativo"].value_counts().to_dict())
# ejemplos representativos por tipo solicitado
for tt in ["negativo→neutro", "neutro→positivo", "positivo→neutro", "negativo→positivo"]:
    sub = err[err["tipo_error"] == tt].head(2)
    if len(sub): print(f"\n-- {tt} --")
    for _, r in sub.iterrows():
        print(f"   [{r.aspecto} | conf {r.confianza:.2f} | {r.patron_tentativo}] {str(r.text_clean)[:120]}")

# ---------- 4) Gráficos ----------
res = pd.read_csv(ac.REP / f"resumen_modelos_{ac.VER}.csv").set_index("modelo")
# bootstrap dist
plt.figure(figsize=(7,4)); plt.hist(boot["f1_macro"], bins=40, color="#4c72b0", alpha=.85)
lo, hi = np.percentile(boot["f1_macro"], [2.5, 97.5])
plt.axvline(lo, ls="--", color="red"); plt.axvline(hi, ls="--", color="red"); plt.axvline(np.mean(boot["f1_macro"]), color="black")
plt.title(f"Bootstrap F1-macro (XLM-R) — IC95% [{lo:.3f}, {hi:.3f}]"); plt.xlabel("F1-macro"); plt.tight_layout(); plt.savefig(OUT/"bootstrap_f1_macro.png", dpi=160); plt.close()
# F1 por clase BERT vs XLM-R
x = np.arange(3); w=.38
fx = [res.loc["xlmr", f"ensemble_f1_{l}"] for l in L]; fb = [res.loc["bert", f"ensemble_f1_{l}"] for l in L]
plt.figure(figsize=(7,4)); plt.bar(x-w/2, fb, w, label="BERT", color="#bbb"); plt.bar(x+w/2, fx, w, label="XLM-R", color="#4c72b0")
plt.xticks(x, L); plt.ylim(0,1); plt.title("F1 por clase: BERT vs XLM-R"); plt.legend(); plt.tight_layout(); plt.savefig(OUT/"f1_por_clase.png", dpi=160); plt.close()
# F1 por aspecto BERT vs XLM-R
ax = pd.read_csv(ac.REP/f"por_aspecto_xlmr_{ac.VER}.csv").rename(columns={"f1_macro":"xlmr"})[["aspecto","xlmr"]]
ab = pd.read_csv(ac.REP/f"por_aspecto_bert_{ac.VER}.csv").rename(columns={"f1_macro":"bert"})[["aspecto","bert"]]
cmp = ab.merge(ax, on="aspecto").sort_values("xlmr"); y=np.arange(len(cmp))
plt.figure(figsize=(8,5)); plt.barh(y-w/2, cmp.bert, w, label="BERT", color="#bbb"); plt.barh(y+w/2, cmp.xlmr, w, label="XLM-R", color="#4c72b0")
plt.yticks(y, cmp.aspecto); plt.xlim(0,1); plt.title("F1 por aspecto: BERT vs XLM-R"); plt.legend(); plt.tight_layout(); plt.savefig(OUT/"f1_por_aspecto.png", dpi=160); plt.close()
# confusion XLM-R
cm = confusion_matrix(yt, yp_x, labels=L); plt.figure(figsize=(5,4)); plt.imshow(cm, cmap="Blues")
plt.xticks(range(3), L, rotation=45); plt.yticks(range(3), L)
for i in range(3):
    for j in range(3): plt.text(j,i,cm[i,j],ha="center",va="center")
plt.title("Matriz de confusión (XLM-R)"); plt.ylabel("real"); plt.xlabel("pred"); plt.tight_layout(); plt.savefig(OUT/"confusion_xlmr.png", dpi=160); plt.close()
# tipos de error
et = err["tipo_error"].value_counts(); plt.figure(figsize=(8,4)); plt.bar(et.index, et.values, color="#d62728")
plt.xticks(rotation=30, ha="right"); plt.title("Tipos de error más frecuentes (XLM-R)"); plt.tight_layout(); plt.savefig(OUT/"tipos_error.png", dpi=160); plt.close()
print("\nGráficos guardados en", OUT)

# ---------- 5) Reporte ----------
lines = ["# Rigor estadístico — Fase 2 (análisis posterior, sin re-entrenar)\n",
    "> No es tuning: no se re-entrena, no se cambian hiperparámetros ni se toca el test. Solo inferencia con los checkpoints v4 guardados + análisis estadístico.\n",
    "## 1. Intervalos de confianza (bootstrap, B=2000) — XLM-R\n",
    "| Métrica | Puntual | IC 95% |", "|---|---|---|"]
for _, r in ci.iterrows(): lines.append(f"| {r.metrica} | {r.puntual} | [{r.IC95_inf}, {r.IC95_sup}] |")
lines += ["", "El IC del F1-macro mide la **incertidumbre** de la estimación (test n=610); no se usa para mover umbrales.", "",
    "## 2. Test de McNemar (XLM-R vs BERT, mismo test pareado)\n",
    f"- Contingencia: ambos correctos {n11} · **XLM-R sí / BERT no {b}** · BERT sí / XLM-R no {c} · ambos incorrectos {n00}",
    f"- Discordantes: {b+c} · χ²(cc)={stat:.3f} · **p (exacto binomial) = {p_exact:.4g}**",
    f"- **Interpretación:** {mc['interpretacion']}.", "- Compara errores pareados en el mismo conjunto de prueba.", "",
    "## 3. Análisis cualitativo de errores\n",
    f"- Errores de XLM-R: **{len(err)}** de {len(test)} ({len(err)/len(test)*100:.1f}%).",
    f"- Por tipo: {err['tipo_error'].value_counts().to_dict()}",
    f"- Patrones tentativos (heurísticos, requieren revisión humana): {err['patron_tentativo'].value_counts().to_dict()}",
    "- Ejemplos en `errores_cualitativos.csv` (texto, destino, aspecto, real, pred, confianza, patrón).",
    "- Los patrones son una **clasificación tentativa automática**; no se editaron etiquetas ni resultados tras ver los errores.", "",
    "## 4. Gráficos", "- `bootstrap_f1_macro.png` · `f1_por_clase.png` · `f1_por_aspecto.png` · `confusion_xlmr.png` · `tipos_error.png`", "",
    "## 5. Conclusión",
    f"El F1-macro de XLM-R (0.709) tiene IC95% [{ci.iloc[0].IC95_inf}, {ci.iloc[0].IC95_sup}]. "
    + ("La mejora sobre BERT está respaldada por McNemar (p<0.05)." if (p_exact < 0.05 and b > c) else "La diferencia con BERT no alcanza significancia en McNemar.")
    + " El análisis de errores se concentra en la frontera negativo↔neutro y aspectos débiles, consistente con las limitaciones declaradas (R20)."]
(OUT / "reporte_rigor_estadistico_fase2.md").write_text("\n".join(lines), encoding="utf-8")
print("Reporte:", OUT / "reporte_rigor_estadistico_fase2.md")
