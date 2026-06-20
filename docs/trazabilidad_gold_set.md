# Trazabilidad y construcción del gold set (Fase 1 → Fase 2)

> Documento de metodología listo para el capítulo correspondiente de la tesis.
> Resume el flujo de datos y la construcción **iterativa** del gold set, de modo que
> el proceso se lea como un refinamiento guiado por evidencia y no como una serie de
> repeticiones.

## 1. Rol de cada fase

La **Fase 1 no produce el gold set final**, sino los **insumos** de los que el gold
set nace y luego se refina en la Fase 2. Concretamente, la Fase 1 entrega:

1. **Corpus limpio**: reseñas de Google Maps de 10 centros turísticos (español e
   inglés), depuradas y deduplicadas (de 36 829 reseñas crudas a **12 657** reseñas
   del corpus ABSA, aplicando un umbral mínimo de longitud derivado empíricamente).
2. **Léxico de aspectos**: el diccionario bilingüe que detecta, por reglas, cuáles de
   los 10 aspectos turísticos se mencionan en cada reseña.
3. **Base de anotación** (`gold_set_base_para_anotacion.csv`, 445 filas): la
   *plantilla/semilla* estratificada desde la que arranca la anotación humana.

La **Fase 2** construye el **gold set** (de forma iterativa) y, sobre él, entrena y
evalúa el modelo ABSA que produce la **matriz destino-aspecto-sentimiento** que
consumirá la Fase 3.

## 2. El módulo ABSA es un flujo híbrido (declaración metodológica)

La extracción de aspectos la realiza un **diccionario validado (etapa basada en
reglas)**, mientras que la **clasificación de polaridad** la realiza un **modelo
profundo**. Por tanto, el proceso ABSA **no es completamente neuronal** y así se
declara: *"extracción de aspectos mediante diccionario + clasificación de polaridad
mediante modelo Transformer"*.

## 3. Construcción iterativa del gold set

El gold set se construyó de forma **incremental**, y **cada iteración respondió a una
limitación detectada por el análisis de errores**, no a un reintento arbitrario:

| Iteración | Qué se hizo | Justificación (evidencia que la motivó) | Acuerdo (κ) | Tamaño |
|---|---|---|---|---|
| Base (Fase 1) | Plantilla de anotación estratificada | Semilla del corpus para iniciar la anotación | — | 445 |
| Gold consenso | Anotación por 3 jueces + etiqueta de consenso | Construir un gold confiable con acuerdo medido | **0.92** | 2 866 |
| **Gold v3** | Ampliación dirigida de **negativos** | El análisis por clase mostró que la **clase negativa** era el cuello de botella | **0.89** | 3 562 |
| **Gold v4** | Refuerzo de **aspectos débiles** (clima, aforo, limpieza) | El desempeño **por aspecto** mostró aspectos con baja señal | **0.82** | 4 045 |

Cada ítem del gold set conserva una columna `origen` (p. ej. `consenso_v2`,
`refuerzo_aspectos_v4`) que permite **trazar de qué iteración proviene** cada ejemplo.

### Efecto del refuerzo de aspectos (v3 → v4)

| Aspecto | v3 (neg/neu/pos) | v4 (neg/neu/pos) |
|---|---|---|
| clima | 65 / 30 / 28 | 125 / 290 / 62 |
| aforo_multitudes | 153 / 105 / 28 | 161 / 110 / 74 |
| limpieza | 74 / 72 / 152 | 89 / 109 / 170 |
| alojamiento | 5 / 83 / 17 | 5 / 83 / 17 *(sin cambio)* |

## 4. Diseño experimental (sobre cualquier versión del gold set)

- **Partición 70/15/15** por `review_uid` (no por fila), con **verificación explícita
  de cero fuga** entre train/validation/test.
- **≥ 3 semillas** y reporte de **media ± desviación estándar** (la selección del
  modelo no se basa en la mejor corrida).
- **Selección de hiperparámetros por validación**, nunca mirando el test.

## 5. Cómo comparar versiones (advertencia metodológica)

Las versiones del gold set (v3, v4) **no son comparables de forma estricta** entre sí,
porque cada una se **re-particiona** y por tanto tiene un **conjunto de test distinto**.
La comparación entre versiones debe presentarse como **referencia de evolución**, no
como una prueba estricta en el mismo test. La mejora real se sostiene con:

- el **desempeño por aspecto** (especialmente los aspectos reforzados), y
- la **estabilidad por semillas** (desviación estándar del F1-macro).

## 6. Limitaciones declaradas

- **`alojamiento`** permanece con baja señal **por límite del corpus**: el corpus
  completo contiene apenas ~282 menciones del aspecto (los centros son atractivos, no
  hoteles), por lo que no puede balancearse mediante más anotación. Se declara como
  limitación; la matriz lo marca con nivel de evidencia *insuficiente/baja* y el
  recomendador lo penaliza por confianza.
- La extracción de aspectos por diccionario implica un **sesgo de cobertura**: las
  menciones que el léxico no captura no entran al estudio. Una extracción neuronal de
  aspectos se plantea como trabajo futuro.

## 7. Redacción sugerida (párrafo modelo)

> "El gold set se construyó de forma incremental a partir de la base de anotación de
> la Fase 1. Tras un conjunto inicial anotado por tres jueces (Fleiss κ = 0.92), el
> análisis de errores reveló dos limitaciones de señal —la clase negativa y un grupo
> de aspectos poco representados—, que se atendieron mediante ampliaciones dirigidas
> (v3: refuerzo de negativos, κ = 0.89; v4: refuerzo de clima, aforo y limpieza,
> κ = 0.82). Cada ampliación conserva trazabilidad de origen y respeta la misma
> taxonomía de 10 aspectos y las mismas etiquetas de polaridad. El módulo ABSA es un
> flujo híbrido: la detección de aspectos se realiza con un diccionario validado y la
> clasificación de polaridad con un modelo Transformer, lo que se declara explícitamente
> para no atribuir todo el proceso a la red neuronal."
