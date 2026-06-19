# -*- coding: utf-8 -*-
# Construye el notebook 03 (bitacora metodologica del entrenamiento gold v3).
# El notebook CARGA los artefactos generados por entrenar_bert_textcnn_v3.py
# (RUN_TRAINING=False) para mostrar resultados sin re-entrenar; el codigo del
# modelo/entrenamiento queda documentado y es reproducible si se pone True.
import json
from pathlib import Path

cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})

def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})

md(r"""
# 03 — Entrenamiento ABSA (BERT multilingual + TextCNN) sobre gold set v3

**Bitácora metodológica.** Este notebook documenta, de forma visible y reproducible,
el entrenamiento y evaluación del módulo ABSA de la Fase 2 sobre el **gold set v3**
(ampliado priorizando la clase negativa). No es una copia del script: es el registro
de *qué datos se usaron, qué se entrenó, qué resultó y si cumple la especificación*.

- **Especificación:** [`specs/modulo-absa-fase2.md`](../specs/modulo-absa-fase2.md)
- **Entrenamiento (motor):** `scripts/entrenar_bert_textcnn_v3.py`
- **Partición:** `scripts/particionar_gold_v3.py` → `outputs/reports/split_report_gold_v3.md`

**Objetivo (criterio de éxito de la spec):** F1-macro ≥ 0.70 en test anotado, con
mínimos por clase (negativo F1 ≥ 0.60 y recall ≥ 0.60; neutro F1 ≥ 0.60). Si no se
alcanza, se entrega como *versión base defendible* con comparación, análisis de
errores y límites del gold set.

> **Nota de ejecución:** por defecto `RUN_TRAINING = False`: el notebook **carga los
> artefactos** ya producidos por el script de entrenamiento (varias horas en GPU).
> Cámbialo a `True` solo si quieres re-entrenar desde el notebook.
""")

code(r"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)

# Raiz del proyecto (este notebook vive en notebooks/)
BASE = Path.cwd().parent if Path.cwd().name.lower() == "notebooks" else Path.cwd()
DATA = BASE / "data"
REP  = BASE / "outputs" / "reports"
VIS  = BASE / "outputs" / "visualizations"

LABELS = ["negativo", "neutro", "positivo"]

# Si True, re-entrena desde el notebook (lento). Si False, carga artefactos del script.
RUN_TRAINING = False

def load_csv(path, **kw):
    path = Path(path)
    if not path.exists():
        print(f"[pendiente] aún no existe: {path.name} (ejecuta scripts/entrenar_bert_textcnn_v3.py)")
        return None
    return pd.read_csv(path, **kw)

print("BASE:", BASE)
""")

md(r"""
## 1. Carga del gold set v3 y de los splits

Se usan el gold set v3 (3.562 ítems aspecto-texto-label, consolidado por consenso de
3 anotadores, κ≈0.89) y las particiones `train/val/test_gold_v3.csv` generadas con
estratificación por polaridad y agrupamiento por `review_uid`.
""")

code(r"""
gold  = pd.read_csv(DATA / "gold_set_v3.csv", encoding="utf-8-sig")
train = pd.read_csv(DATA / "train_gold_v3.csv", encoding="utf-8-sig")
val   = pd.read_csv(DATA / "val_gold_v3.csv", encoding="utf-8-sig")
test  = pd.read_csv(DATA / "test_gold_v3.csv", encoding="utf-8-sig")
for n in (gold, train, val, test):
    n["label"] = n["label"].astype(str).str.lower().str.strip()

print(f"gold v3 : {len(gold):5d} ítems | {gold['review_uid'].nunique()} reseñas únicas")
print(f"train   : {len(train):5d} ({len(train)/len(gold)*100:.1f}%)")
print(f"val     : {len(val):5d} ({len(val)/len(gold)*100:.1f}%)")
print(f"test    : {len(test):5d} ({len(test)/len(gold)*100:.1f}%)")
train.head(3)
""")

md(r"""
## 2. Verificación de no fuga por `review_uid`

Requisito crítico: ninguna reseña puede aparecer en más de un split (evita que el
modelo "vea" en entrenamiento texto que luego evalúa). Se comprueba la intersección
de `review_uid` entre los tres splits.
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

Auditoría para confirmar que los splits son representativos y comparables.
""")

code(r"""
def tabla(col):
    t = (pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
           .pivot_table(index=col, columns="split", values="label", aggfunc="count", fill_value=0))
    return t.reindex(columns=["train", "val", "test"], fill_value=0)

print("Polaridad por split (conteo):")
display(tabla("label"))
print("\nPolaridad por split (%):")
display((tabla("label") / tabla("label").sum() * 100).round(1))
print("\nAspecto por split:");  display(tabla("aspecto"))
if "language_review" in train: print("\nIdioma por split:"); display(tabla("language_review"))
if "destination" in train:     print("\nDestino por split:"); display(tabla("destination"))
""")

code(r"""
# Visual: proporción de polaridad por split (debe ser casi igual entre splits)
prop = (tabla("label") / tabla("label").sum() * 100)[["train", "val", "test"]]
ax = prop.T.plot(kind="bar", figsize=(7, 4))
ax.set_ylabel("% de ítems"); ax.set_title("Distribución de polaridad por split (gold v3)")
ax.legend(title="polaridad"); plt.xticks(rotation=0); plt.tight_layout(); plt.show()
""")

md(r"""
## 4. Modelo: BERT multilingual + TextCNN

Arquitectura base oficial del PPI: un encoder Transformer multilingüe
(`bert-base-multilingual-cased`) cuyas representaciones por token pasan por
convoluciones 1D (kernels 2/3/4) + max-pooling, y una capa lineal de clasificación
en 3 clases. La ablación previa mostró que **cambiar el encoder no es la palanca**;
por eso se conserva esta arquitectura y se ataca la señal de la clase negativa.
""")

code(r"""
import torch, torch.nn as nn

MODEL_NAME = "bert-base-multilingual-cased"
MAX_LEN, BATCH, EPOCHS = 256, 4, 12
LR, WEIGHT_DECAY, WARMUP_RATIO, PATIENCE, DROPOUT = 2e-5, 0.10, 0.10, 3, 0.40
CNN_FILTERS, CNN_KERNELS = 128, (2, 3, 4)

class BERTTextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        h = self.bert.config.hidden_size
        self.convs = nn.ModuleList([nn.Conv1d(h, CNN_FILTERS, k) for k in CNN_KERNELS])
        self.drop = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(CNN_FILTERS * len(CNN_KERNELS), 3)
    def forward(self, ids, mask):
        x = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state.transpose(1, 2)
        pooled = [torch.max(torch.relu(c(x)), dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(pooled, dim=1)))

print("Config:", dict(model=MODEL_NAME, max_len=MAX_LEN, batch=BATCH, epochs=EPOCHS,
                       lr=LR, dropout=DROPOUT, cnn_filters=CNN_FILTERS, cnn_kernels=CNN_KERNELS))
""")

md(r"""
## 5. Configuración de entrenamiento enfocada a la clase negativa

La clase negativa es el cuello de botella (la ablación lo mostró). Dos mecanismos:

- **Class weights** inversamente proporcionales a la frecuencia, con un **refuerzo
  extra** a la clase negativa (`NEG_BOOST = 1.5`).
- **Focal Loss** (`gamma = 2.0`): reduce el peso de los ejemplos fáciles y enfoca el
  aprendizaje en los difíciles (típicamente negativos confundidos con neutro).

Abajo se muestran los **pesos reales** que recibe cada clase con el corpus de train.
""")

code(r"""
FOCAL_GAMMA = 2.0
NEG_BOOST = 1.5

class FocalLoss(nn.Module):
    def __init__(self, weight, gamma): super().__init__(); self.w=weight; self.g=gamma
    def forward(self, logits, y):
        ce = nn.functional.cross_entropy(logits, y, weight=self.w, reduction="none")
        return (((1 - torch.exp(-ce)) ** self.g) * ce).mean()

def class_weights(labels):
    c = pd.Series(labels).value_counts().reindex(LABELS, fill_value=0); tot = c.sum()
    w = [tot / (len(LABELS) * c[l]) if c[l] > 0 else 0.0 for l in LABELS]
    w[LABELS.index("negativo")] *= NEG_BOOST
    return w

w = class_weights(train["label"])
print("Pesos por clase (con refuerzo a negativa):")
for l, x in zip(LABELS, w): print(f"  {l:9s}: {x:.3f}")
print(f"Focal gamma = {FOCAL_GAMMA} | NEG_BOOST = {NEG_BOOST}")
""")

md(r"""
## 6. Entrenamiento con ≥ 3 semillas

El entrenamiento real (3 semillas: 42, 7, 123; early stopping por F1-macro en
validación; ensemble por promedio de probabilidades) lo ejecuta
`scripts/entrenar_bert_textcnn_v3.py`, que **genera los artefactos** que este
notebook carga. Re-ejecutar dentro del notebook toma varias horas en GPU, por eso
`RUN_TRAINING = False` por defecto.
""")

code(r"""
if RUN_TRAINING:
    import subprocess, sys
    print("Re-entrenando desde el notebook (lento)...")
    subprocess.run([sys.executable, str(BASE / "scripts" / "entrenar_bert_textcnn_v3.py")], check=True)
else:
    print("RUN_TRAINING = False -> se cargan los artefactos de scripts/entrenar_bert_textcnn_v3.py")
""")

md(r"""
## 7. Resultados por semilla

Una fila por semilla: F1-macro, accuracy, y F1/recall por clase en el **test**.
""")

code(r"""
det = load_csv(REP / "resultados_bert_textcnn_v3.csv")
display(det)
""")

md(r"""
## 8. Promedio ± desviación estándar

La selección del modelo no se basa en la mejor corrida sino en el **promedio** sobre
semillas. Una desviación estándar alta del F1-macro (> 0.03) indica inestabilidad.
""")

code(r"""
if det is not None:
    cols = [c for c in det.columns if c != "seed"]
    agg = pd.DataFrame({"media": det[cols].mean().round(4), "desv_std": det[cols].std().round(4)})
    display(agg)
    std_macro = det["f1_macro"].std()
    print(f"Desv. estándar F1-macro = {std_macro:.4f} -> {'ESTABLE' if std_macro <= 0.03 else 'INESTABLE (>0.03)'}")
""")

md(r"""
## 9. Métricas obligatorias

F1-macro, F1 por clase, recall negativo, matriz de confusión y desempeño por aspecto
(sobre el **ensemble** de las semillas).
""")

code(r"""
cr = load_csv(REP / "classification_report_v3.csv", index_col=0)
if cr is not None:
    print("Classification report (ensemble):"); display(cr.round(3))

asp = load_csv(REP / "por_aspecto_v3.csv")
if asp is not None:
    print("\nF1-macro por aspecto (ensemble), de peor a mejor:"); display(asp)
""")

code(r"""
from IPython.display import Image
cm_path = VIS / "matriz_confusion_v3.png"
Image(str(cm_path)) if cm_path.exists() else print("[pendiente] matriz_confusion_v3.png")
""")

md(r"""
## 10. Veredicto automático contra la especificación

- **Éxito técnico:** F1-macro ≥ 0.70 **y** negativo F1 ≥ 0.60 **y** recall negativo ≥ 0.60 **y** neutro F1 ≥ 0.60.
- **Versión base defendible:** si no alcanza el umbral, pero entrega comparación vs.
  baselines, análisis de errores y límites del gold set (que este pipeline produce).
""")

code(r"""
res = load_csv(REP / "resumen_bert_textcnn_v3.csv")
if res is not None:
    r = res.iloc[0]
    checks = {
        "F1-macro ≥ 0.70":      (r["ensemble_f1_macro"], 0.70),
        "Negativo F1 ≥ 0.60":   (r["ensemble_f1_negativo"], 0.60),
        "Recall negativo ≥ 0.60": (r["ensemble_recall_negativo"], 0.60),
        "Neutro F1 ≥ 0.60":     (r["ensemble_f1_neutro"], 0.60),
    }
    print("=== Veredicto contra la spec (ensemble) ===")
    todos = True
    for k, (val_, th) in checks.items():
        ok = val_ >= th; todos &= ok
        print(f"  [{'PASA' if ok else 'FALLA'}] {k:24s} -> {val_:.3f}")
    estable = r.get("VEREDICTO_estable_std<=0.03", None)
    print(f"\nEstabilidad (std F1-macro ≤ 0.03): {estable}")
    print("\nRESULTADO:", "✅ ÉXITO TÉCNICO (cumple la spec)" if todos
          else "⚠️ NO alcanza el umbral -> VERSIÓN BASE DEFENDIBLE (con evidencia metodológica)")
""")

md(r"""
## 11. Exportación de resultados (trazabilidad)

Todos los artefactos quedan en `outputs/reports/` y `outputs/visualizations/`. Aquí se
consolida un índice JSON para trazabilidad.
""")

code(r"""
artefactos = {
    "split_report": "outputs/reports/split_report_gold_v3.md",
    "resultados_por_semilla": "outputs/reports/resultados_bert_textcnn_v3.csv",
    "resumen_y_veredicto": "outputs/reports/resumen_bert_textcnn_v3.csv",
    "classification_report": "outputs/reports/classification_report_v3.csv",
    "por_aspecto": "outputs/reports/por_aspecto_v3.csv",
    "matriz_confusion": "outputs/visualizations/matriz_confusion_v3.png",
    "gold_set_v3": "data/gold_set_v3.csv",
    "splits": ["data/train_gold_v3.csv", "data/val_gold_v3.csv", "data/test_gold_v3.csv"],
}
with open(REP / "indice_trazabilidad_v3.json", "w", encoding="utf-8") as f:
    json.dump(artefactos, f, ensure_ascii=False, indent=2)
print("Índice de trazabilidad -> outputs/reports/indice_trazabilidad_v3.json")
for k, v in artefactos.items(): print(f"  {k}: {v}")
""")

md(r"""
## 12. Conclusión

<!-- CONCLUSION_PLACEHOLDER -->
*(Esta sección se completa con los números finales una vez termina el entrenamiento.)*

**Qué se hizo.** Se entrenó el módulo ABSA (BERT multilingual + TextCNN) sobre el
gold set v3 (3.562 ítems, partición 70/15/15 por `review_uid` sin fuga), con una
configuración enfocada a la clase negativa (focal loss γ=2.0 + class weights con
refuerzo ×1.5 a negativa) y 3 semillas.

**Lectura metodológica.** El resultado se interpreta contra la especificación: si
alcanza F1-macro ≥ 0.70 con los mínimos por clase, es un *éxito técnico*; si no, es
una *versión base defendible* sustentada por la comparación, el análisis de errores y
las limitaciones declaradas del gold set (p. ej. la escasez intrínseca de negativos en
`alojamiento` y `gastronomia`, limitada por el corpus).
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).resolve().parent.parent / "notebooks" / "03_entrenamiento_absa_bert_textcnn_gold_v3.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook escrito:", out, "| celdas:", len(cells))
