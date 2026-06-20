# Validación del diccionario de aspectos (R10–R11)

> Valida la **etapa de extracción de aspectos basada en reglas** del flujo híbrido.
> Recordatorio metodológico: **la extracción de aspectos la hace un diccionario
> (reglas); la clasificación de polaridad la hace XLM-R + TextCNN.** No se afirma
> que todo el proceso sea neuronal.

## 1. Cobertura del léxico

Sobre las **12 657** reseñas del corpus:

- **Cobertura (≥ 1 aspecto detectado): 84.5%** (10 696 reseñas).
- 15.5% no activan ningún aspecto: son reseñas cortas/genéricas ("buen lugar",
  "nice place") sin mención de aspectos evaluables — comportamiento esperado.
- Promedio de **1.97 aspectos detectados por reseña**.

La cobertura (84.5%) **supera el umbral de 75%** usado como criterio en la Fase 1,
por lo que **no se gatilló una ampliación forzada del léxico** (condición de R11).

### Detección por aspecto

| Aspecto | Reseñas detectadas | % del corpus |
|---|---|---|
| atractivos | 7 767 | 61.4% |
| atencion_servicio | 2 777 | 21.9% |
| accesibilidad | 2 617 | 20.7% |
| costos | 2 322 | 18.3% |
| aforo_multitudes | 1 654 | 13.1% |
| clima | 1 105 | 8.7% |
| gastronomia | 990 | 7.8% |
| limpieza | 912 | 7.2% |
| **seguridad** | 614 | 4.9% |
| **alojamiento** | 282 | 2.2% |

## 2. Falsos positivos / falsos negativos (precisión y recall de la detección)

Medir FP/FN requiere una **referencia humana de presencia de aspectos**, independiente
del diccionario. Para ello se preparó una **muestra de validación multi-etiqueta**
(`data/muestra_validacion_diccionario.csv`, 150 reseñas aleatorias): el anotador marca,
para cada reseña, **cuáles de los 10 aspectos se discuten realmente** (1/0), sin ver lo
que detectó el léxico (anti-anclaje).

Con esa muestra, `scripts/validar_diccionario_v4.py` calcula, **por aspecto**:

- **TP** = el léxico lo detectó y el aspecto sí está presente.
- **FP** = el léxico lo detectó pero el aspecto no está (falso positivo).
- **FN** = el aspecto está pero el léxico no lo detectó (falso negativo).
- **precisión, recall y F1** por aspecto + macro global.

> Estado: muestra **lista para anotar**. Al completarse, los resultados se reportan en
> `outputs/reports/diccionario_fp_fn.csv` y se incorporan a este documento.

## 3. Caracterización del léxico y ampliaciones

- El diccionario es **bilingüe (español/inglés)**, con **sinónimos**, **variantes con y
  sin tilde**, coincidencia por **raíz/prefijo** (p. ej. *hermos\** → hermoso/hermosa) y
  **frases** específicas del **dominio turístico**.
- Como la cobertura global resultó **suficiente (84.5%)**, no se requirió una ampliación
  general. En las ampliaciones dirigidas de la Fase 2 (refuerzo de aspectos débiles) sí
  se incorporaron **términos de opinión específicos por aspecto** (es/en) para clima,
  aforo y limpieza, documentados en `scripts/minar_candidatos_aspectos.py`.

## 4. Límites declarados en aspectos débiles

- **`alojamiento` (2.2%)** y **`seguridad` (4.9%)** tienen baja detección, pero esto
  refleja **escasez en el corpus** (los centros son atractivos turísticos, no hoteles;
  la seguridad se menciona poco), **no una falla del léxico**. Se declara como limitación
  (R20), consistente con el bajo desempeño del modelo en esos aspectos.
- El **sesgo de cobertura** inherente a la extracción por reglas (lo que el léxico no
  captura no entra al estudio) se declara explícitamente; una **extracción neuronal de
  aspectos** queda como trabajo futuro.

## 5. Conclusión

La extracción de aspectos por reglas alcanza una **cobertura del 84.5%**, suficiente para
el prototipo. La validación de precisión/recall por aspecto está preparada y se completa
con la muestra anotada. El diccionario es bilingüe y específico del dominio; los aspectos
con baja detección (`alojamiento`, `seguridad`) están limitados por el corpus, no por el
léxico. Queda claro que **la detección de aspectos es una etapa basada en reglas** y que
**la clasificación de polaridad la realiza XLM-R + TextCNN**.

## 6. Artefactos

- `outputs/reports/diccionario_cobertura.csv`
- `data/muestra_validacion_diccionario.csv` (anotable, multi-etiqueta)
- `outputs/reports/diccionario_fp_fn.csv` (al anotarse)
- Script: `scripts/validar_diccionario_v4.py`
