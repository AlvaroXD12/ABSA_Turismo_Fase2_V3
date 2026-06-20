# PIPELINE — Fase 1 → Fase 2 → Fase 3

Pipeline completo del sistema ABSA para recomendación turística, con **entradas y
salidas por etapa**. La Fase 2 es el alcance de este repositorio; las Fases 1 y 3 se
describen para dar contexto.

## Visión general

```
Fase 1 (corpus)  →  Fase 2 (ABSA: gold + modelo + matriz)  →  Fase 3 (recomendador)
```

---

## Fase 1 — Corpus y reseñas

**Qué hace:** construye el corpus limpio a partir de reseñas crudas de Google Maps.

- **Entrada:** archivos JSON del scraper (Google Maps, 10 centros turísticos, es/en).
- **Proceso:** estandarización → anonimización → limpieza → filtro de idioma/longitud →
  deduplicación → umbral empírico de longitud (14 palabras) → léxico de aspectos.
- **Salida:**
  - corpus ABSA limpio (`tourism_reviews_clean.csv`, ~12 657 reseñas),
  - **diccionario de aspectos** (10 aspectos, bilingüe),
  - base de anotación (semilla del gold set).

Detalle: `docs/fase1_construccion_corpus.md`.

---

## Fase 2 — ABSA (alcance de este repo)

### 2.1 Gold set (construcción iterativa)

**Qué hace:** anota y refina el conjunto de referencia.

- **Entrada:** corpus + base de anotación de la Fase 1.
- **Proceso:** anotación por 3 jueces + consenso (kappa); ampliaciones dirigidas por
  análisis de errores (v3: refuerzo de negativos; v4: refuerzo de clima/aforo/limpieza).
- **Salida:** `data/gold_set_v4.csv` (4 045 ítems aspecto-texto-label).
- **Scripts:** `minar_candidatos_*.py`, `consolidar_*.py`, `preparar_lotes_*.py`.
- Detalle: `docs/trazabilidad_gold_set.md`.

### 2.2 Diccionario de aspectos (validación)

**Qué hace:** valida la etapa de extracción por reglas.

- **Entrada:** corpus + léxico de aspectos.
- **Proceso/Salida:** cobertura (~84.5%), y plantilla para FP/FN por aspecto.
- **Script:** `validar_diccionario_v4.py`. Detalle: `docs/validacion_diccionario_R10_R11.md`.

### 2.3 Partición

**Qué hace:** divide el gold set sin fuga.

- **Entrada:** `gold_set_v4.csv`.
- **Proceso:** split 70/15/15 **por `review_uid`** (sin fuga), estratificado por polaridad.
- **Salida:** `data/{train,val,test}_gold_v4.csv` + `outputs/reports/split_report_gold_v4.md`.
- **Script:** `particionar_gold_v4.py`.

### 2.4 Entrenamiento y evaluación ABSA

**Qué hace:** entrena y evalúa los modelos.

- **Entrada:** splits del gold v4.
- **Proceso:** XLM-R + TextCNN (principal) y BERT + TextCNN (base), 5 semillas,
  focal loss + class weights, **selección de HP en validación**, calibración de decisión,
  ensemble; baseline TF-IDF + LogReg.
- **Salida:** métricas por semilla, media±std, por clase, por aspecto, matriz de confusión,
  veredicto vs spec (`outputs/reports/*.csv`).
- **Notebook:** `03_entrenamiento_absa_xlmr_textcnn_gold_v4.ipynb`.
- **Scripts (opción B):** `entrenar_modelo_v4.py`, `absa_common.py`.

### 2.5 Evaluación secundaria y rigor

- **Secundaria (distribución real):** `evaluacion_secundaria_v4.py` — coherencia cuando
  predominan positivas. Detalle: `docs/eval_secundaria_R17_R18.md`.
- **Rigor estadístico:** `rigor_estadistico_fase2.py` — IC bootstrap, McNemar, análisis de
  errores. Salida: `outputs/rigor_estadistico/`.

### 2.6 Matriz destino-aspecto-sentimiento

**Qué hace:** produce el insumo de la Fase 3.

- **Entrada:** predicciones del modelo final sobre el corpus completo.
- **Proceso:** agrega por (destino × aspecto); calcula conteos, proporciones,
  `sentiment_score`, `score_ajustado`, `dominant_label`, `evidence_status`, `confidence`,
  `conflict_flag`.
- **Salida:** `outputs/matrices/matriz_destino_aspecto_sentimiento.{csv,json}` (100 celdas).
- **Script:** `generar_matriz_absa.py`.

---

## Fase 3 — Recomendador (fuera de este repo)

**Qué hace:** consume la matriz para recomendar de forma explicable.

- **Entrada:** `matriz_destino_aspecto_sentimiento.csv` + perfil del viajero.
- **Proceso:** modelo de **afinidad destino-perfil** (pondera aspectos según preferencias)
  + capa de **explicación** (usa `score_ajustado`, `evidence_status` y `confidence` para
  decir *con cuánta evidencia* y *por qué aspecto* se sostiene la recomendación).
- **Salida:** ranking de destinos personalizado y explicado.
