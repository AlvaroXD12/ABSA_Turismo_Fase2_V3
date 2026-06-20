# Resultados — Fase 2 (módulo ABSA)

Resultados sobre el **test estratificado del gold v4** (610 instancias, partición
70/15/15 por `review_uid`, sin fuga). Las métricas de los modelos profundos son el
**ensemble calibrado** de **5 semillas**. La métrica principal es **F1-macro**.

## 1. Comparación de modelos

| Modelo | F1-macro (test) |
|---|---|
| TF-IDF + Logistic Regression (baseline) | 0.574 |
| BERT multilingual + TextCNN (base oficial PPI) | 0.651 |
| **XLM-R + TextCNN (modelo final)** | **0.709** |

Progresión clara: el baseline clásico queda lejos; XLM-R supera a la base oficial.
La mejora de XLM-R sobre BERT es **estadísticamente significativa** (ver §4).

## 2. Métricas por clase (XLM-R, modelo final)

| Clase | F1 | Recall | Mínimo spec | ¿Cumple? |
|---|---|---|---|---|
| negativo | 0.636 | 0.651 | F1 ≥ 0.60 y recall ≥ 0.60 | ✅ |
| neutro | 0.736 | 0.705 | F1 ≥ 0.60 | ✅ |
| positivo | 0.755 | 0.700 | — | — |
| **macro** | **0.709** | — | ≥ 0.70 | ✅ |

Accuracy = 0.721. Comparativa por clase vs BERT en `outputs/rigor_estadistico/f1_por_clase.png`.
Matriz de confusión: `outputs/rigor_estadistico/confusion_xlmr.png`.

## 3. Estabilidad por semillas (5 semillas)

| Modelo | F1-macro media ± std | ¿Estable (std ≤ 0.03)? |
|---|---|---|
| XLM-R + TextCNN | 0.691 ± 0.016 | ✅ |
| BERT + TextCNN | 0.646 ± 0.009 | ✅ |

El resultado **no depende de una corrida afortunada**: el F1-macro es estable entre
semillas. El valor reportado (0.709) es el **ensemble** de las 5.

## 4. Intervalos de confianza y significancia (análisis posterior, sin re-entrenar)

**Bootstrap (B=2000), XLM-R:**

| Métrica | Puntual | IC 95% |
|---|---|---|
| F1-macro | 0.708 | [0.672, 0.745] |
| F1 negativo | 0.635 | [0.566, 0.701] |
| Recall negativo | 0.651 | [0.564, 0.731] |
| F1 neutro | 0.735 | [0.693, 0.777] |

> Lectura honesta: el F1-macro **puntual cruza 0.70**, pero con n=610 el IC 95% baja a
> 0.672. Se reporta así por rigor; no se usa para mover umbrales.

**McNemar XLM-R vs BERT (errores pareados, mismo test):**
- Contingencia: ambos correctos 375 · **XLM-R sí / BERT no: 65** · BERT sí / XLM-R no: 34 · ambos incorrectos 136.
- **p ≈ 0.0024 → la mejora de XLM-R sobre BERT es estadísticamente significativa.**

Distribución bootstrap: `outputs/rigor_estadistico/bootstrap_f1_macro.png`.

## 5. Desempeño por aspecto (XLM-R)

| Aspecto | F1-macro | Aspecto | F1-macro |
|---|---|---|---|
| limpieza | 0.71 | atractivos | 0.54 |
| seguridad | 0.66 | clima | 0.53 |
| atencion_servicio | 0.56 | gastronomia | 0.47 |
| accesibilidad | 0.56 | aforo_multitudes | 0.46 |
| costos | 0.56 | **alojamiento** | **0.29** |

Los aspectos reforzados en v4 mejoraron (p. ej. limpieza 0.43→0.71). Comparativa
v3 vs v4 y BERT vs XLM-R en `outputs/rigor_estadistico/f1_por_aspecto.png`.

**Análisis de errores** (170/610 = 27.9%): se concentran en la frontera **↔neutro**
(positivo→neutro 63, negativo→neutro 37). Patrones tentativos: ambigüedad semántica,
reseñas mixtas, aspecto débil. Ejemplos en `outputs/rigor_estadistico/errores_cualitativos.csv`.

## 6. Matriz destino-aspecto-sentimiento

Producto final de la Fase 2 (insumo de la Fase 3): **100 celdas** (10 destinos × 10
aspectos), con `sentiment_score`, `score_ajustado`, `dominant_label`, `evidence_status`,
`confidence` y `conflict_flag`.

- Niveles de evidencia: 95 *suficiente*, 4 *insuficiente*, 1 *baja*; 3 celdas en conflicto.
- **A diferencia de un proxy por estrellas (que daría casi todo positivo), la matriz
  discrimina sentimiento por aspecto.** Ej. Machu Picchu: `aforo_multitudes` = −0.54
  (negativo), `clima` = −0.51 (negativo), `atractivos` = +0.97 (positivo), `seguridad`
  = mixta/conflictiva (confianza 0.65).
- Archivo: `outputs/matrices/matriz_destino_aspecto_sentimiento.{csv,json}`.

## 7. Limitaciones

- **Aspectos con bajo soporte:** `alojamiento` (F1 0.29) y `aforo_multitudes` (0.46)
  están limitados por la **escasez del corpus** (p. ej. alojamiento ~282 menciones; las
  multitudes casi nunca se elogian), **no por el modelo**. Declarados como limitación.
- **Frontera negativo↔neutro:** principal fuente de error, propia de la subjetividad de
  la tarea.
- **Extracción de aspectos por reglas:** sesgo de cobertura (~84.5%); una extracción
  neuronal queda como trabajo futuro.
- **Comparación entre versiones del gold (v3/v4):** no es estricta (tests distintos); se
  usa solo como referencia de evolución.
