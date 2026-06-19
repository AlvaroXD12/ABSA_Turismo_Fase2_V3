# Guía de anotación — Ampliación gold set v2 (foco en negativos)

## Qué etiquetas
Para cada fila, etiqueta la **polaridad del ASPECTO indicado** (columna `aspecto`)
tal como aparece en la reseña (`text_clean`). **No** etiquetas la reseña completa
ni la calificación de estrellas: solo lo que el texto dice **sobre ese aspecto**.

Escribe en la columna `label` exactamente una de: `positivo`, `neutro`, `negativo`.

## Definiciones (léelas para evitar el "neutro de escape")
- **negativo** — el texto expresa crítica, queja, insatisfacción o problema sobre
  el aspecto. Cuenta como negativo **aunque la reseña en general sea positiva**.
  Ej.: "Lugar hermoso, pero **los baños estaban sucios**" → para `limpieza` es
  **negativo**.
- **positivo** — el texto elogia o expresa satisfacción clara sobre el aspecto.
  Ej.: "la **entrada es muy barata** para lo que ofrece" → para `costos` es
  **positivo**.
- **neutro** — el aspecto se menciona de forma **descriptiva/factual sin
  valoración** clara, o hay una valoración **genuinamente equilibrada** dentro del
  mismo aspecto (algo bueno y algo malo que se compensan). Ej.: "la entrada
  cuesta 30 soles" (solo dato) → `costos` **neutro**.

## Reglas para no abusar del neutro
- "neutro" **no** es "no estoy seguro" ni "no sé". Si dudas, vuelve a leer qué
  dice el texto sobre el aspecto y decide entre positivo/negativo; usa neutro solo
  cuando de verdad no hay valoración o está equilibrada.
- Una crítica leve **sigue siendo negativa**, no neutra (ej. "un poco caro",
  "algo descuidado").
- No conviertas en neutro un aspecto solo porque la reseña general es positiva.

## Casos especiales
- Si el aspecto **no se discute realmente** en el texto (la palabra clave
  apareció pero no se opina del aspecto), deja `label` vacío y, si puedes, anota
  "no aplica" en una nota. Estas filas se descartan luego.
- El idioma puede ser español o inglés; etiqueta igual en ambos.

## Procedimiento
1. Cada anotador trabaja su archivo `muestra_anotador_<n>.csv`.
2. Una parte (120 items) está en los tres archivos a propósito: es el
   **solapamiento** para medir el acuerdo entre anotadores (kappa). Anótenlos
   con normalidad, sin compararse entre ustedes.
3. No modifiquen `annotation_id`, `review_uid` ni `aspecto`. Solo llenan `label`.
4. Al terminar, devuelvan los tres archivos para consolidación, cálculo de kappa
   y construcción de la nueva partición.
