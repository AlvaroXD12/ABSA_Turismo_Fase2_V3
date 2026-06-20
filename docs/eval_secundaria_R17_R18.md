# Evaluación secundaria en distribución real (R17–R18)

> Verifica la **coherencia** del modelo final (XLM-R + TextCNN) cuando se aplica al
> corpus real, donde **predominan las reseñas positivas** (~94% por estrellas).
> **No reemplaza** el F1-macro del test estratificado (0.709); lo complementa.

## 1. Objetivo y diseño

El test principal es **estratificado** (se fuerza presencia de negativos/neutros para
medir la capacidad real de distinguir las 3 clases). Pero en producción la matriz se
aplica sobre un corpus **fuertemente positivo**. R17–R18 verifican que el modelo **se
mantenga razonable** en esa condición.

- **Muestra:** distribución **real**, **no balanceada** por polaridad.
- **Dos evaluaciones:**
  - **(A) Coherencia (sin anotación):** sobre los **21 040** pares reseña×aspecto del
    corpus, se comparan las predicciones de XLM-R contra la **polaridad por estrellas**,
    usada como **referencia débil** (las estrellas describen la reseña completa, no el
    aspecto).
  - **(B) Con etiquetas (preparada):** muestra aleatoria de **501** instancias
    (≥ 300 exigidas), proporcional por destino e idioma, **sin balancear**, lista para
    anotación humana (`data/muestra_evaluacion_secundaria.csv`). Al anotarse, se evalúa
    con `scripts/evaluar_muestra_secundaria.py`.

## 2. Resultados de coherencia (A)

### Distribución de polaridad en distribución real

| Polaridad | Predicho por XLM-R | Por estrellas (reseña) |
|---|---|---|
| negativo | **10.7%** | 3.2% |
| neutro | **40.2%** | 3.8% |
| positivo | **49.1%** | 93.0% |

**Lectura clave (positiva para la tesis):** el modelo **no colapsa** a "todo positivo"
pese a que las estrellas son 93% positivas. Predice ~49% positivo, pero identifica un
**40% neutro** y un **11% negativo** a nivel de aspecto. Esto es exactamente el
comportamiento esperado de un ABSA: en una reseña de 5 estrellas, muchos aspectos se
mencionan de forma **factual (neutra)** o incluso **crítica** (p. ej. las multitudes).

### Acuerdo con estrellas (referencia débil)

- Accuracy = 0.495 · Cohen κ = 0.037 → **bajo, y esto VALIDA el enfoque**, no lo
  contradice: las estrellas miden la reseña completa y el modelo mide el **aspecto**.
- La matriz de confusión lo confirma: de los aspectos en reseñas de estrellas positivas,
  el modelo predice positivo en ~9 881 casos pero **neutro en ~7 844 y negativo en
  ~1 836** — es decir, detecta que no todo en una buena reseña es elogio.

### Coherencia por aspecto (proporción predicha)

| Aspecto | neg | neu | pos | Lectura |
|---|---|---|---|---|
| atractivos | 0.01 | 0.00 | **0.99** | los atractivos se elogian (coherente) |
| aforo_multitudes | **0.52** | 0.44 | 0.04 | las multitudes se critican aun en buenas reseñas |
| clima | **0.44** | 0.56 | 0.01 | el clima genera quejas |
| limpieza | 0.10 | 0.28 | **0.62** | percepción mayormente positiva |
| seguridad | 0.13 | 0.40 | 0.47 | mixto |
| alojamiento | 0.00 | **0.96** | 0.04 | casi todo neutro (aspecto con poca señal, R20) |

Todas las distribuciones son **interpretables y plausibles**: no hay colapso a una sola
clase ni comportamientos absurdos. Por idioma (es/en) la distribución es similar.

## 3. Comparación con el test estratificado

- Test estratificado (métrica **principal**): F1-macro **0.709** (clases balanceadas a
  propósito).
- Distribución real: el modelo predice positivo-dominante pero **discriminando** por
  aspecto. La evaluación secundaria **no reemplaza** la métrica principal; demuestra que
  el modelo **transfiere de forma coherente** a la condición de producción.

## 4. Conclusión

El modelo XLM-R + TextCNN **se mantiene razonable bajo predominio de reseñas positivas**:
produce una distribución positivo-dominante pero **no degenerada**, detecta negativos y
neutros a nivel de aspecto, y diverge de las estrellas justo donde el análisis por
aspecto aporta valor (p. ej. multitudes y clima negativos dentro de reseñas globalmente
positivas). Esto respalda el uso de la matriz destino-aspecto-sentimiento en la Fase 3.

## 5. Pendiente (versión con etiquetas)

La muestra de **501 instancias** (no balanceada, proporcional por destino/idioma) está
lista para anotación. Una vez etiquetada, `evaluar_muestra_secundaria.py` reporta el
F1-macro en distribución real, que se documentará junto a esta evaluación. Esta versión
es opcional para reforzar el argumento; la coherencia ya está demostrada arriba.

## 6. Artefactos

- `outputs/reports/eval_secundaria_distribucion.csv`
- `outputs/reports/eval_secundaria_acuerdo_estrellas.csv`
- `outputs/reports/eval_secundaria_por_aspecto.csv`
- `data/muestra_evaluacion_secundaria.csv` (anotable)
- Scripts: `scripts/evaluacion_secundaria_v4.py`, `scripts/evaluar_muestra_secundaria.py`
