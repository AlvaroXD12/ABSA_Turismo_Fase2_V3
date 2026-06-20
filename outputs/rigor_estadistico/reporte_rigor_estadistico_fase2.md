# Rigor estadístico — Fase 2 (análisis posterior, sin re-entrenar)

> No es tuning: no se re-entrena, no se cambian hiperparámetros ni se toca el test. Solo inferencia con los checkpoints v4 guardados + análisis estadístico.

## 1. Intervalos de confianza (bootstrap, B=2000) — XLM-R

| Métrica | Puntual | IC 95% |
|---|---|---|
| f1_macro | 0.7082 | [0.6718, 0.7452] |
| f1_negativo | 0.6353 | [0.566, 0.7007] |
| recall_negativo | 0.6507 | [0.5639, 0.7308] |
| f1_neutro | 0.7353 | [0.6926, 0.7768] |

El IC del F1-macro mide la **incertidumbre** de la estimación (test n=610); no se usa para mover umbrales.

## 2. Test de McNemar (XLM-R vs BERT, mismo test pareado)

- Contingencia: ambos correctos 375 · **XLM-R sí / BERT no 65** · BERT sí / XLM-R no 34 · ambos incorrectos 136
- Discordantes: 99 · χ²(cc)=9.091 · **p (exacto binomial) = 0.002395**
- **Interpretación:** XLM-R mejora a BERT de forma estadísticamente significativa (p<0.05).
- Compara errores pareados en el mismo conjunto de prueba.

## 3. Análisis cualitativo de errores

- Errores de XLM-R: **170** de 610 (27.9%).
- Por tipo: {'positivo→neutro': 63, 'negativo→neutro': 37, 'positivo→negativo': 28, 'neutro→negativo': 23, 'neutro→positivo': 11, 'negativo→positivo': 8}
- Patrones tentativos (heurísticos, requieren revisión humana): {'ambiguedad_semantica': 73, 'aspecto_debil': 68, 'resena_mixta': 28, 'corta_falta_contexto': 1}
- Ejemplos en `errores_cualitativos.csv` (texto, destino, aspecto, real, pred, confianza, patrón).
- Los patrones son una **clasificación tentativa automática**; no se editaron etiquetas ni resultados tras ver los errores.

## 4. Gráficos
- `bootstrap_f1_macro.png` · `f1_por_clase.png` · `f1_por_aspecto.png` · `confusion_xlmr.png` · `tipos_error.png`

## 5. Conclusión
El F1-macro de XLM-R (0.709) tiene IC95% [0.6718, 0.7452]. La mejora sobre BERT está respaldada por McNemar (p<0.05). El análisis de errores se concentra en la frontera negativo↔neutro y aspectos débiles, consistente con las limitaciones declaradas (R20).