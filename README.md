# ABSA Turismo — Fase 2: análisis de sentimiento por aspectos

Módulo de **Análisis de Sentimiento Basado en Aspectos (ABSA)** para un sistema de
recomendación de centros turísticos del Perú. Esta Fase 2 entrena y evalúa el modelo
ABSA que clasifica la polaridad (positivo / neutro / negativo) de cada aspecto
turístico, y produce la **matriz destino-aspecto-sentimiento** que consumirá la Fase 3
(modelo de afinidad destino-perfil y explicación).

## Objetivo y contexto

El turista encuentra información dispersa en reseñas que no está organizada por
criterios comparables. Este módulo transforma reseñas públicas (Google Maps, es/en) en
**señales estructuradas por aspecto**, para que la app pueda recomendar de forma
personalizada y **explicable**.

## Arquitectura metodológica (flujo híbrido)

> **Extracción de aspectos por diccionario (reglas)** + **clasificación de polaridad
> con XLM-R + TextCNN (modelo profundo).** Se declara explícitamente: la detección de
> aspectos es una etapa basada en reglas; la red neuronal solo clasifica polaridad.

- **Encoder:** XLM-RoBERTa (`xlm-roberta-base`) — candidato final.
- **Cabezal:** TextCNN (convoluciones 1D, kernels 2/3/4 + max-pooling).
- **Base oficial de comparación del PPI:** BERT multilingual + TextCNN.
- **Baseline clásico:** TF-IDF + Logistic Regression.

### Taxonomía de 10 aspectos
`atractivos · costos · seguridad · accesibilidad · limpieza · atencion_servicio ·
gastronomia · alojamiento · clima · aforo_multitudes`

## Estructura del repositorio

```
notebooks/   Notebooks finales (pipeline reproducible)
scripts/     Scripts ejecutables (gold set, entrenamiento, evaluación, matriz)
data/        Splits del gold set y muestras anotables (insumos ligeros)
outputs/     Métricas, matrices, predicciones, reportes y gráficos generados
docs/        Documentación metodológica (pipeline, resultados, trazabilidad)
models/      Checkpoints entrenados (NO versionados; ver .gitignore)
archive/     Material histórico: notebooks viejos, experimentos, dev (no es pipeline)
```

## Notebooks finales

| Notebook | Rol |
|---|---|
| `01_preparacion_goldset_y_diccionario.ipynb` | Construcción del gold set + acuerdo entre anotadores (kappa) |
| `02_validacion_absa_y_baselines.ipynb` | Validación ABSA y baselines (versión previa, referencia) |
| **`03_entrenamiento_absa_xlmr_textcnn_gold_v4.ipynb`** | **Notebook principal**: entrenamiento + evaluación + matriz |

## Datos de entrada esperados

- `data/gold_set_v4.csv` y splits `data/{train,val,test}_gold_v4.csv` (70/15/15, por `review_uid`).
- `outputs/predictions/tourism_reviews_clean_absa_ready.csv` (corpus con aspectos detectados, para la matriz).

## Cómo ejecutar el pipeline

```bash
pip install -r requirements.txt          # instala PyTorch según tu CUDA (ver requirements.txt)
```

**Opción A — notebook principal (todo en uno):**
abrir `notebooks/03_entrenamiento_absa_xlmr_textcnn_gold_v4.ipynb` → *Restart & Run All*.
Entrena (5 semillas) XLM-R y BERT, calibra, evalúa, compara y genera la matriz.

**Opción B — entrenamiento por modelo en procesos separados (GPU de poca VRAM):**
```bash
python scripts/entrenar_modelo_v4.py --model xlmr
python scripts/entrenar_modelo_v4.py --model bert
```

**Análisis estadístico posterior (sin re-entrenar):**
```bash
python scripts/rigor_estadistico_fase2.py     # IC bootstrap, McNemar, análisis de errores
```

## Salidas generadas

- **Métricas:** `outputs/reports/*.csv` (por semilla, media±std, por clase, por aspecto).
- **Matriz destino-aspecto-sentimiento:** `outputs/matrices/matriz_destino_aspecto_sentimiento.{csv,json}`.
- **Predicciones:** `outputs/predictions/`.
- **Rigor estadístico + gráficos:** `outputs/rigor_estadistico/`.
- **Reportes y documentación:** `docs/`.

## Resultado final

**XLM-R + TextCNN sobre el gold v4 alcanza F1-macro = 0.709** en el test estratificado,
cumpliendo los mínimos por clase (negativo F1 ≥ 0.60 y recall ≥ 0.60; neutro F1 ≥ 0.60)
y de forma **estable** (desv. estándar entre semillas ≤ 0.03). La mejora sobre BERT es
**estadísticamente significativa** (McNemar, p ≈ 0.002). Ver `docs/RESULTADOS_FASE2.md`.

## Limitaciones

- **Aspectos con bajo soporte** (`alojamiento`, `aforo_multitudes`, en parte `seguridad`):
  limitados por la **escasez del corpus**, no por el modelo. Declarados como limitación.
- **Dependencia del diccionario:** la extracción de aspectos por reglas implica un sesgo
  de cobertura (lo que el léxico no captura no entra). Cobertura medida: ~84.5%.
- Una **extracción neuronal de aspectos** queda como trabajo futuro.

## Reglas de reproducibilidad

- **No tocar el conjunto de test** ni ajustar hiperparámetros mirándolo (la selección de
  `NEG_BOOST`/`FOCAL_GAMMA` se hace en **validación**).
- **No re-entrenar para inflar métricas**; los resultados se reportan como **media ±
  desviación estándar** sobre semillas fijas.
- **Splits versionados** (por `review_uid`, sin fuga) y **semillas documentadas**.
- **Rutas relativas** (los notebooks/scripts resuelven la raíz del proyecto solos; no hay
  rutas absolutas de una PC).

## Documentación

- `docs/PIPELINE.md` — pipeline completo Fase 1 → 2 → 3 (entradas/salidas por etapa).
- `docs/RESULTADOS_FASE2.md` — resultados detallados (modelos, métricas, matriz, límites).
- `docs/fase1_construccion_corpus.md` · `docs/trazabilidad_gold_set.md`
- `docs/eval_secundaria_R17_R18.md` · `docs/validacion_diccionario_R10_R11.md`
