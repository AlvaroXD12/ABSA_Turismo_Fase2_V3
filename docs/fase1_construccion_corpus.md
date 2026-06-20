# Fase 1 — Construcción del corpus turístico (ETL, limpieza y consolidación)

> Documento de metodología listo para el capítulo correspondiente de la tesis.
> Describe qué hace la Fase 1: convertir reseñas crudas de Google Maps en un
> **corpus limpio, anonimizado y deduplicado**, apto para el análisis de
> sentimiento por aspectos (ABSA) de la Fase 2.

## 1. Objetivo

La Fase 1 **no entrena ningún modelo**. Su objetivo es construir la materia prima
de todo el estudio:

1. un **corpus limpio** de reseñas (español e inglés) de 10 centros turísticos del
   Perú;
2. un **léxico de aspectos** (diccionario bilingüe) para la detección de aspectos
   por reglas;
3. la **base de anotación** (semilla del gold set) que consume la Fase 2.

## 2. Fuente de datos

- **Origen:** reseñas públicas de **Google Maps**, extraídas con un scraper
  (Apify), en formato JSON.
- **Cobertura:** **10 centros turísticos** del Perú, en **español e inglés**.
- **Volumen inicial:** **20 archivos JSON** → **36 829 reseñas crudas**.
- El nombre de cada archivo permite inferir el centro turístico y el idioma de
  extracción; una **tabla maestra** define el nombre formal, región y tipo de
  experiencia de cada centro (trazabilidad).

## 3. Pipeline ETL y estandarización

Cada reseña cruda se estandariza a un esquema común con campos de texto,
calificación (estrellas), idioma, identificadores y metadatos del lugar. Se
conserva `source_file` y `source_row_number` para **trazabilidad** hasta la fila
original.

## 4. Anonimización

El corpus final **no conserva datos del usuario/reviewer**. Se eliminan todas las
columnas personales (nombre, usuario, identificadores, URL de perfil, foto, etc.)
antes de cualquier exportación.

## 5. Limpieza y normalización de texto

- Normalización **mínima compatible con BERT**: se eliminan saltos de línea y URLs
  y se colapsan espacios, **conservando tildes y mayúsculas/minúsculas** (no se
  eliminan stopwords ni se hace stemming), porque el modelo Transformer aprovecha
  esa información.
- **Conteo de palabras** ignorando URLs (para los criterios de longitud).
- **Firma textual** (texto en minúsculas, sin URLs ni signos) para deduplicar por
  contenido.
- La polaridad por estrellas se deriva como referencia: 1–2 = negativo, 3 =
  neutro, 4–5 = positivo.

## 6. Filtro base de relevancia → "crudo controlado"

Se descartan reseñas **vacías**, con **menos de 1 palabra**, o en **idiomas
distintos de español/inglés**. El resultado es el **crudo controlado**.

- 36 829 crudas → **25 139** reseñas únicas en el crudo controlado.

> El umbral de longitud para ABSA **no** se aplica aquí: se determina
> empíricamente en el paso 8.

## 7. Deduplicación cruzada

Una misma reseña puede aparecer dos veces (recuperada en el archivo de español y
en el de inglés). Se deduplica en dos niveles:

1. por **`reviewId`** (cuando existe);
2. por **firma textual dentro del mismo destino**.

En empates se **prioriza** la fila cuyo idioma de la reseña coincide con el idioma
de extracción y la versión **no traducida**.

## 8. Umbral de longitud para ABSA (determinación empírica)

El mínimo de palabras para que una reseña entre al corpus ABSA **no se fija a
mano**, se **estima empíricamente** sobre el corpus mediante dos criterios:

- **Curva de cobertura de aspectos por longitud:** para reseñas con ≥ X palabras se
  mide qué porcentaje menciona al menos un aspecto turístico evaluable; el umbral
  es el primer X donde la cobertura cruza el **75%**.
- **Punto de codo (saturación):** donde la curva deja de crecer.

El umbral resultante es de **14 palabras**. Aplicándolo:

- crudo controlado (25 139) → **corpus ABSA = 12 657** reseñas.

## 9. Léxico de aspectos (diccionario bilingüe)

La Fase 1 define el **diccionario de aspectos** (español/inglés) que detecta, por
reglas, cuáles de los **10 aspectos turísticos canónicos** se mencionan en cada
reseña:

> atractivos · costos · seguridad · accesibilidad · limpieza ·
> atencion_servicio · gastronomia · alojamiento · clima · aforo_multitudes

Este léxico es la **etapa basada en reglas** del flujo híbrido (extracción de
aspectos por diccionario + clasificación de polaridad por modelo profundo en la
Fase 2).

## 10. Reportes de calidad y matriz de decisión de centros

Se generan reportes por **destino, fuente e idioma** (volumen, % en inglés, % de
textos cortos, distribución de estrellas) y una **matriz de decisión** que marca,
por centro, si cumple volumen total, mínimo de reseñas en inglés y densidad
textual suficiente, clasificándolo en *mantener / observar / reemplazar*. Estas
tablas respaldan con evidencia la selección de centros.

## 11. Salidas de la Fase 1

- **`tourism_reviews_clean.csv`** — corpus ABSA (12 657 reseñas, ≥ 14 palabras),
  insumo directo de la Fase 2.
- Corpus general consolidado (todas las reseñas únicas, sin el umbral de 14).
- **Base de anotación** del gold set (semilla estratificada, 445 filas).
- Reportes de calidad, matriz de decisión, visualizaciones y log de ejecución.

## 12. Caracterización del corpus final

| Eje | Distribución |
|---|---|
| Idioma | español 8 345 · inglés 4 312 |
| Polaridad por estrellas | positivo 11 851 · neutro 471 · **negativo 335** |
| Longitud | media ≈ 47 palabras · mediana ≈ 33 |
| Centros (top) | Machu Picchu 3 135 · Circuito Mágico del Agua 2 086 · Centro Histórico de Lima 1 648 · Huaca Pucllana 1 356 · Monasterio de Santa Catalina 1 030 · … |

**Hallazgo clave:** el corpus está **fuertemente sesgado a positivo** (~94% de las
reseñas son 4–5 estrellas; solo ~2.6% son negativas). Este desbalance es una
característica real del dominio (las reseñas turísticas tienden a ser positivas) y
**motivó el muestreo estratificado y las ampliaciones dirigidas del gold set en la
Fase 2** (refuerzo de la clase negativa y de aspectos poco representados).

## 13. Limitaciones de la Fase 1

- **Sesgo positivo del corpus:** limita la cantidad de ejemplos negativos
  disponibles para entrenar/evaluar; se gestiona en la Fase 2, pero algunos
  aspectos (p. ej. `alojamiento`) quedan intrínsecamente poco representados.
- **Detección de aspectos por diccionario:** introduce un **sesgo de cobertura**
  (las menciones que el léxico no captura no entran al estudio); se declara
  explícitamente y una extracción neuronal de aspectos se plantea como trabajo
  futuro.
- **Dependencia de la calidad del scraping** (idioma mal etiquetado, reseñas
  traducidas), mitigada con la deduplicación con prioridad por idioma/no-traducción.

## 14. Redacción sugerida (párrafo modelo)

> "La Fase 1 construyó el corpus a partir de 36 829 reseñas crudas de Google Maps
> de 10 centros turísticos del Perú (español e inglés). Tras estandarizar,
> anonimizar y limpiar el texto (normalización compatible con Transformers,
> conservando tildes), se filtraron reseñas vacías o en otros idiomas (25 139
> reseñas únicas) y se deduplicó por identificador y firma textual con prioridad
> por idioma de extracción. El umbral mínimo de longitud para ABSA (14 palabras)
> se determinó empíricamente mediante la curva de cobertura de aspectos y su punto
> de saturación, obteniendo un corpus final de 12 657 reseñas. La Fase 1 también
> definió el diccionario bilingüe de 10 aspectos turísticos (etapa de extracción
> por reglas del flujo híbrido) y la base de anotación que alimenta la Fase 2. El
> corpus resultante está fuertemente sesgado a opiniones positivas (~94%), lo que
> motivó el muestreo estratificado del gold set en la fase siguiente."
