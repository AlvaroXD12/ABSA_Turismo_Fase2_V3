# Especificación: Módulo ABSA (Fase 2) — PPI v2

## Objetivo
Mejorar y dejar metodológicamente sólido el **módulo ABSA (Fase 2)** del PPI v2:
clasificar la polaridad (positivo / neutro / negativo) de los 10 aspectos
turísticos de cada reseña y producir una **matriz analítica
destino-aspecto-sentimiento** confiable que sirva de insumo a la Fase 3 (modelo
de afinidad destino-perfil y capa de explicación). El módulo debe alcanzar un
umbral de desempeño definido o, en su defecto, quedar como versión base
defendible con evidencia experimental honesta.

## Contexto
El PPI v2 se replantea como un aplicativo móvil de recomendación de centros
turísticos del Perú basado en análisis de sentimiento por aspectos. El módulo
ABSA es la Fase 2: convierte reseñas públicas (Google Maps, es+en) en señales
estructuradas por aspecto. Hoy el notebook 02 (BERT multilingual + TextCNN)
obtiene ~0.62 de F1-macro (bajo el umbral de 0.70), mejora poco frente al mejor
baseline clásico, y tiene clases/aspectos flojos. Además, la extracción de
aspectos no la hace el modelo profundo sino un **diccionario de palabras clave**:
el flujo real es **híbrido** (reglas para extraer aspecto + modelo profundo para
polaridad) y así debe declararse. Esta spec define qué debe cumplir el módulo
para considerarse terminado y confiable como insumo de la Fase 3.

Taxonomía fija de 10 aspectos: atractivos, costos, seguridad, accesibilidad,
limpieza, atencion_servicio, gastronomia, alojamiento, clima, aforo_multitudes.
Etiquetas de polaridad fijas: positivo, neutro, negativo.

## Requisitos

### Desempeño del modelo
1. **[Imprescindible]** El modelo final debe alcanzar **F1-macro ≥ 0.70** sobre
   el **test anotado manualmente y estratificado** del gold set.
2. **[Imprescindible]** Mínimos por clase en ese mismo test:
   - clase **negativo**: **F1 ≥ 0.60** y **recall ≥ 0.60**;
   - clase **neutro**: **F1 ≥ 0.60**.
   Estos mínimos existen para que el promedio macro no oculte que la negativa o
   el neutro fallan (el neutro no debe usarse como "categoría de escape").
3. **[Imprescindible]** El desempeño reportado debe provenir del **promedio de
   ≥ 3 semillas** (ver R7), no de una única corrida favorable.

### Protocolo experimental y selección de arquitectura
4. **[Imprescindible]** Implementar **BERT multilingual + TextCNN** como
   arquitectura **base y punto de comparación oficial**. No es obligatorio que
   sea el modelo final.
5. **[Imprescindible]** Se pueden evaluar variantes (p. ej. XLM-RoBERTa, BETO,
   RoBERTa-es, modelos preentrenados en sentimiento, cambios de cabezal,
   hiperparámetros, balanceo de clases, ampliación del gold set, ensembles)
   **siempre bajo el mismo protocolo**: misma taxonomía de aspectos, mismas
   etiquetas de polaridad y **las mismas particiones train/validation/test**.
6. **[Imprescindible]** El **modelo final se selecciona por desempeño medible**,
   priorizando en este orden: F1-macro, recall de la clase negativa y
   estabilidad por aspecto — usando el **promedio** sobre semillas, no la mejor
   corrida individual.
7. **[Imprescindible]** **Estabilidad por semillas:** los modelos candidatos
   principales se evalúan con **≥ 3 semillas** (mismas particiones) y se reporta
   **media ± desviación estándar** de: F1-macro, F1 negativo, recall negativo,
   F1 neutro y desempeño por aspecto. Si la **desv. estándar del F1-macro > 0.03**,
   el modelo se marca como **inestable** y no puede presentarse como superior de
   forma concluyente.

### Evidencia metodológica (obligatoria, alcance o no el umbral)
8. **[Imprescindible]** Producir, para el modelo final y la base:
   **comparación honesta vs. baselines** (incluida la clase mayoritaria como
   piso), **matriz de confusión**, **análisis de errores**, **desempeño por
   clase** y **desempeño por aspecto**.
9. **[Imprescindible]** Declarar explícitamente el flujo como **híbrido**:
   extracción de aspectos por **diccionario validado (reglas)** + clasificación
   de polaridad por **modelo profundo**. La redacción del PPI debe ser:
   *"modelo Transformer para ABSA con comparación de variantes, usando BERT
   multilingual + TextCNN como arquitectura base"*. No afirmar que todo el
   proceso ABSA es neuronal.

### Validación del diccionario de aspectos
10. **[Imprescindible]** Validar el diccionario sobre una **muestra anotada
    manualmente**, midiendo: **cobertura del léxico**, **falsos negativos**,
    **falsos positivos** y **desempeño por aspecto**.
11. **[Imprescindible]** Si la cobertura resulta insuficiente, **ampliar el
    léxico** con sinónimos, expresiones frecuentes, variantes es/en y términos
    del dominio turístico, y re-medir.

### Matriz destino-aspecto-sentimiento (producto principal)
12. **[Imprescindible]** Granularidad principal **destino × aspecto**, agregando
    todas las reseñas válidas (es + en) y todas las fuentes consideradas.
13. **[Imprescindible]** Cada celda (destino × aspecto) debe contener **todos**
    estos campos:
    - `n_menciones` (total de menciones del aspecto),
    - `n_resenas_unicas` (reseñas únicas que respaldan la celda),
    - `n_positivo`, `n_neutro`, `n_negativo` (conteos de predicciones),
    - `prop_positivo`, `prop_neutro`, `prop_negativo` (proporciones),
    - `score_sentimiento` (escala −1..1, ver fórmula),
    - `etiqueta_dominante` (solo interpretativa; el score y las proporciones son
      las variables principales para la Fase 3),
    - `nivel_evidencia`, `conflict_flag`, `confianza`.
14. **[Imprescindible]** **Score de sentimiento** (−1..1):
    `score_sentimiento = (n_positivo - n_negativo) / n_total`
    (donde `n_total = n_positivo + n_neutro + n_negativo`). ~1 = percepción
    positiva; ~−1 = negativa; ~0 = neutra, mixta o débilmente definida.
15. **[Imprescindible]** **Confianza de la celda**:
    `confidence_evidencia = min(1, n_menciones / 10)`;
    `confianza = confidence_evidencia * 0.65` si `conflict_flag = 1`,
    de lo contrario `confianza = confidence_evidencia`.
15b. **[Imprescindible]** **Regla de conflicto.** `conflict_flag = 1` cuando se
    cumplen **todas** estas condiciones: `n_menciones ≥ 5`, `prop_positivo ≥ 0.25`,
    `prop_negativo ≥ 0.25` y `|prop_positivo − prop_negativo| < 0.15`. En ese
    caso la `etiqueta_dominante` se fija en "mixta/conflictiva" (no se fuerza a
    neutro) y la Fase 3 debe tratar la celda como **evidencia dividida**.
16. **[Imprescindible]** **Variables auxiliares** para auditoría/trazabilidad y
    análisis de sesgos: desagregación por **idioma**, **fuente** y **fecha de
    actualización**. La Fase 3 consume la matriz agregada; el análisis
    metodológico puede desagregar por idioma o fuente cuando sea necesario.

### Evaluación secundaria (coherencia en condiciones de producción)
17. **[Imprescindible]** Además del test principal, evaluar el modelo sobre una
    **muestra aleatoria del corpus real** posterior a la detección de aspectos,
    **no balanceada por polaridad** (refleja el predominio de positivas).
18. **[Imprescindible]** La muestra secundaria tendrá **≥ 300 instancias**
    destino-aspecto-texto (**500 si el corpus lo permite**), conservando
    **proporcionalmente destino, idioma y fuente**, sin alterar artificialmente
    la distribución natural de polaridades. Esta evaluación **no reemplaza** el
    F1-macro del test principal; verifica coherencia.

### Gold set
19. **[Imprescindible]** **Tamaño del gold set ampliado.** Objetivo mínimo
    **2.500 instancias** anotadas aspecto-texto-label; objetivo deseable
    **3.000**. Pisos metodológicos: **≥ 100 instancias por aspecto** y **≥ 500
    instancias por polaridad global**, priorizando la ampliación de **negativos,
    neutros y aspectos con bajo soporte**.
20. **[Imprescindible]** Si **no** se alcanza alguno de esos pisos (2.500 total,
    100 por aspecto, 500 por polaridad), debe **declararse explícitamente como
    limitación del gold set** y reflejarse en el análisis de errores.

### Deseables
21. **[Deseable]** Justificar la **suficiencia del gold set** con evidencia
    adicional (curva de aprendizaje y/o tamaño por fórmula de proporciones).
22. **[Deseable]** Explorar **ensembles** y encoders adicionales si acercan al
    umbral sin romper el protocolo.

## Fuera de alcance
- **Extracción neuronal de aspectos.** Se mantiene la extracción por diccionario;
  un extractor neuronal queda como **mejora futura / variante experimental**, no
  es requisito para cerrar el prototipo.
- **Fase 3 completa** (modelo de afinidad destino-perfil, ranking, capa de
  explicación) y la **app móvil**. Esta spec entrega solo la matriz como insumo.
- **Reemplazo de la taxonomía de aspectos o de las etiquetas de polaridad.**
- **Re-scrapeo o ampliación del corpus crudo** (Fase 1). Solo se usa el corpus
  ya consolidado.

## Restricciones
- **Mismo protocolo entre variantes:** taxonomía de 10 aspectos, 3 etiquetas de
  polaridad y **las mismas particiones train/validation/test** para toda
  comparación. Cambiar particiones entre modelos invalida la comparación.
- **Reproducibilidad:** semillas fijadas y registradas; resultados reportados
  como media ± desv. estándar sobre ≥ 3 semillas.
- **Hardware:** GPU con ~4 GB (GTX 1650 Ti). Las técnicas para entrar en memoria
  (precisión mixta, congelado parcial de capas, batch/longitud ajustados) son
  admisibles siempre que no rompan el protocolo de comparación.
- **Idiomas:** corpus bilingüe español/inglés; el modelo debe manejar ambos.
- **Trazabilidad:** la matriz conserva variables por idioma/fuente/fecha.

## Casos extremos
| Condición | Comportamiento esperado |
|---|---|
| Aspecto **sin ninguna mención** en un destino | Celda `nivel_evidencia = "sin datos"`. **No** se interpreta como neutro; no entra al ranking de Fase 3. |
| **1–4 menciones** | `nivel_evidencia = "evidencia insuficiente"`. No debe influir directamente en el ranking. |
| **5–9 menciones** | `nivel_evidencia = "baja evidencia"`. Usable solo con penalización de confianza. |
| **≥ 10 menciones** | `nivel_evidencia = "evidencia suficiente"`. Puede alimentar el modelo de afinidad. |
| **Conflicto de polaridades** (se cumplen **todas**: `n_menciones ≥ 5`, `prop_positivo ≥ 0.25`, `prop_negativo ≥ 0.25` y `|prop_positivo − prop_negativo| < 0.15`) | `conflict_flag = 1`, etiqueta "mixta/conflictiva". **No** se fuerza a neutro; el sistema comunica que las opiniones están divididas, el score se usa con confianza reducida (×0.65) y la Fase 3 la trata como **evidencia dividida**. |
| Reseña en idioma fuera de es/en | Excluida del corpus válido (no entra a la matriz). |
| Aspecto con **muy pocos ejemplos en el test** | Se reporta su F1 por aspecto pero se advierte la baja muestra; no se concluye superioridad sobre esa base. |
| Diccionario detecta aspecto pero el texto no opina sobre él (falso positivo) | Cuenta como FP en la validación del diccionario (R10); se busca reducir vía ajuste de léxico. |
| Desempeño del modelo **inestable** (std F1-macro > 0.03) | Se marca inestable; no se presenta como superior concluyente (R7). |
| No se alcanza el umbral (R1/R2) | Se entrega como **versión base defendible** con toda la evidencia de R8 y la justificación de límites del gold set. |

## Definición de "hecho"
Checklist verificable. El módulo está "hecho" como **versión exitosa** si se
cumplen todos los ítems marcados (E); como **versión base defendible** si se
cumplen los ítems (B) aunque fallen los de desempeño.

**Desempeño (E):**
- [ ] (E) F1-macro ≥ 0.70 en test anotado, promedio de ≥ 3 semillas. *(R1, R3)*
- [ ] (E) Negativo: F1 ≥ 0.60 y recall ≥ 0.60 (promedio de semillas). *(R2)*
- [ ] (E) Neutro: F1 ≥ 0.60 (promedio de semillas). *(R2)*
- [ ] (E) Desv. estándar del F1-macro ≤ 0.03 (si > 0.03, marcado inestable). *(R7)*

**Protocolo y selección (B + E):**
- [ ] BERT-mult+TextCNN implementado como base oficial. *(R4)*
- [ ] Todas las variantes comparadas con mismas particiones/taxonomía/etiquetas. *(R5)*
- [ ] Modelo final elegido por promedio de desempeño (F1-macro, recall neg,
      estabilidad por aspecto), no por mejor corrida. *(R6)*
- [ ] Reporte media ± std (≥3 semillas) de F1-macro, F1-neg, recall-neg, F1-neu,
      por aspecto. *(R7)*

**Evidencia metodológica (B):**
- [ ] Comparación vs. baselines (incluida clase mayoritaria). *(R8)*
- [ ] Matriz de confusión + análisis de errores + desempeño por clase + por aspecto. *(R8)*
- [ ] Flujo declarado como híbrido y redacción metodológica corregida. *(R9)*
- [ ] Validación del diccionario: cobertura, FN, FP, por aspecto sobre muestra
      anotada; léxico ampliado si la cobertura fue baja. *(R10, R11)*

**Matriz (B):**
- [ ] Matriz destino × aspecto generada con **todos** los campos de R13. *(R12, R13)*
- [ ] `score_sentimiento` calculado con la fórmula de R14. *(R14)*
- [ ] `confianza` calculada con la fórmula de R15 (incluida penalización por conflicto). *(R15)*
- [ ] `conflict_flag` aplicado con la regla exacta de R15b (`n_menciones ≥ 5`,
      `prop_pos ≥ 0.25`, `prop_neg ≥ 0.25`, `|prop_pos − prop_neg| < 0.15`). *(R15b)*
- [ ] Niveles de evidencia aplicados según los casos extremos. *(casos extremos)*
- [ ] Variables auxiliares por idioma/fuente/fecha presentes. *(R16)*

**Gold set (B):**
- [ ] Gold set ampliado a **≥ 2.500** instancias (ideal 3.000). *(R19)*
- [ ] Pisos cumplidos: **≥ 100 por aspecto** y **≥ 500 por polaridad**, o bien
      cualquier piso no alcanzado **declarado como limitación** y reflejado en el
      análisis de errores. *(R19, R20)*

**Evaluación secundaria (B):**
- [ ] Evaluación sobre muestra real (≥300, ideal 500), proporcional por
      destino/idioma/fuente, no balanceada; reportada junto al test principal. *(R17, R18)*

## Preguntas abiertas
Ninguna pendiente que bloquee la implementación. Ambas decisiones previamente
abiertas quedaron cerradas: el tamaño y los pisos del gold set ampliado están
fijados en R19–R20, y la regla de conflicto (incluida la "presencia relevante"
de cada polaridad) está fijada en R15b y en la tabla de casos extremos.
