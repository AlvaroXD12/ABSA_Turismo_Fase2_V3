# Reporte de partición — gold set v3

- **Fuente:** `data/gold_set_v3.csv` — 3562 ítems, 1564 reseñas únicas.
- **Método:** StratifiedGroupKFold (n_splits=20, shuffle, seed=42), agrupado por `review_uid`.
  test = folds {0,1,2}, val = folds {3,4,5}, train = resto. Objetivo 70/15/15.
- **No-fuga por `review_uid`:** ✅ SIN solapamiento (train∩val=0, train∩test=0, val∩test=0).

## Tamaño por split

| split | ítems | % | reseñas únicas |
|---|---|---|---|
| train | 2498 | 70.13 | 1077 |
| val | 532 | 14.94 | 243 |
| test | 532 | 14.94 | 244 |

## Polaridad por split (conteo / %)

| split | negativo | neutro | positivo |
|---|---|---|---|
| train | 545 (21.82%) | 809 (32.39%) | 1144 (45.8%) |
| val | 117 (21.99%) | 173 (32.52%) | 242 (45.49%) |
| test | 116 (21.8%) | 173 (32.52%) | 243 (45.68%) |

### Aspecto

| Aspecto | train | val | test |
|---|---|---|---|
| accesibilidad | 321 | 65 | 55 |
| aforo_multitudes | 198 | 37 | 51 |
| alojamiento | 65 | 19 | 21 |
| atencion_servicio | 387 | 79 | 82 |
| atractivos | 688 | 147 | 145 |
| clima | 79 | 25 | 19 |
| costos | 314 | 56 | 63 |
| gastronomia | 74 | 19 | 12 |
| limpieza | 199 | 55 | 44 |
| seguridad | 173 | 30 | 40 |

### Idioma

| Idioma | train | val | test |
|---|---|---|---|
| en | 1105 | 229 | 230 |
| es | 1393 | 303 | 302 |

### Destino

| Destino | train | val | test |
|---|---|---|---|
| Centro Histórico de Lima | 277 | 40 | 59 |
| Circuito Mágico del Agua | 275 | 91 | 66 |
| Ciudadela de Kuélap | 205 | 71 | 20 |
| Líneas y Geoglifos de Nasca y Palpa | 173 | 42 | 30 |
| Monasterio de Santa Catalina | 253 | 32 | 48 |
| Museo Tumbas Reales del Señor de Sipán | 161 | 20 | 63 |
| Museo de Sitio Huaca Pucllana | 312 | 51 | 82 |
| Reserva Nacional de Paracas | 219 | 57 | 27 |
| Santuario Histórico de Machu Picchu | 459 | 77 | 92 |
| Valle del Colca | 164 | 51 | 45 |

## Notas

- Partición por grupo `review_uid`: ninguna reseña aparece en más de un split (sin fuga).
- El test es NUEVO (derivado del gold v3), no se reutilizó el test anterior.
- La estratificación es aproximada: el agrupamiento por reseña impone pequeños ajustes sobre el 70/15/15 exacto.