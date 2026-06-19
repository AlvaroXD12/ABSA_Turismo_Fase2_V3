# Reporte de partición — gold set v4

- **Fuente:** `data/gold_set_v4.csv` — 4045 ítems, 2007 reseñas únicas.
- **Método:** StratifiedGroupKFold (n_splits=20, shuffle, seed=42), agrupado por `review_uid`.
  test = folds {0,1,2}, val = folds {3,4,5}, train = resto. Objetivo 70/15/15.
- **No-fuga por `review_uid`:** ✅ SIN solapamiento (train∩val=0, train∩test=0, val∩test=0).

## Tamaño por split

| split | ítems | % | reseñas únicas |
|---|---|---|---|
| train | 2826 | 69.86 | 1367 |
| val | 609 | 15.06 | 313 |
| test | 610 | 15.08 | 327 |

## Polaridad por split (conteo / %)

| split | negativo | neutro | positivo |
|---|---|---|---|
| train | 602 (21.3%) | 1017 (35.99%) | 1207 (42.71%) |
| val | 130 (21.35%) | 219 (35.96%) | 260 (42.69%) |
| test | 129 (21.15%) | 221 (36.23%) | 260 (42.62%) |

### Aspecto

| Aspecto | train | val | test |
|---|---|---|---|
| accesibilidad | 311 | 68 | 62 |
| aforo_multitudes | 245 | 45 | 55 |
| alojamiento | 68 | 18 | 19 |
| atencion_servicio | 390 | 73 | 85 |
| atractivos | 703 | 148 | 129 |
| clima | 309 | 87 | 81 |
| costos | 316 | 60 | 57 |
| gastronomia | 73 | 16 | 16 |
| limpieza | 246 | 63 | 59 |
| seguridad | 165 | 31 | 47 |

### Idioma

| Idioma | train | val | test |
|---|---|---|---|
| en | 1215 | 270 | 263 |
| es | 1611 | 339 | 347 |

### Destino

| Destino | train | val | test |
|---|---|---|---|
| Centro Histórico de Lima | 261 | 86 | 68 |
| Circuito Mágico del Agua | 357 | 56 | 76 |
| Ciudadela de Kuélap | 245 | 49 | 38 |
| Líneas y Geoglifos de Nasca y Palpa | 174 | 42 | 41 |
| Monasterio de Santa Catalina | 250 | 51 | 71 |
| Museo Tumbas Reales del Señor de Sipán | 182 | 37 | 50 |
| Museo de Sitio Huaca Pucllana | 365 | 66 | 65 |
| Reserva Nacional de Paracas | 270 | 58 | 49 |
| Santuario Histórico de Machu Picchu | 483 | 120 | 131 |
| Valle del Colca | 239 | 44 | 21 |

## Notas

- Partición por grupo `review_uid`: ninguna reseña aparece en más de un split (sin fuga).
- El test es NUEVO (derivado del gold v4), no se reutilizó el test anterior.
- La estratificación es aproximada: el agrupamiento por reseña impone pequeños ajustes sobre el 70/15/15 exacto.