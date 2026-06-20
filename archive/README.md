# archive/ — material histórico (NO forma parte del pipeline final)

Esta carpeta conserva versiones intermedias, experimentos y herramientas de
desarrollo. **No se eliminan** porque son evidencia metodológica del proceso
iterativo, pero **no se usan en el pipeline final**. El flujo reproducible está en
`notebooks/`, `scripts/` y `docs/` (ver el `README.md` raíz).

## Contenido

### `archive/notebooks/` — notebooks superados
- `03_entrenamiento_absa_bert_textcnn_gold_v3.ipynb` — entrenamiento sobre el gold
  set v3 (versión previa; superada por el gold v4 y XLM-R).
- `03_reporte_absa_xlmr_bert_gold_v4.ipynb` — variante "Opción B" (entrenamiento por
  modelo en procesos separados + notebook-reporte). Útil en GPU de poca VRAM; el
  notebook final all-in-one la reemplaza. Sus scripts (`absa_common.py`,
  `entrenar_modelo_v4.py`) siguen en `scripts/`.

### `archive/experiments/` — experimentos y decisiones justificadas
- `curva_aprendizaje.py` — curva de aprendizaje (justifica el tamaño del gold set).
- `mejoras_modelo_ablacion.py`, `mejoras_ablacion_rapido.py`, `run_cardiff.py` —
  ablación de encoders (mBERT vs XLM-R vs modelo de sentimiento) que motivó elegir
  XLM-R como candidato principal.
- `entrenar_bert_textcnn_v3.py`, `particionar_gold_v3.py` — pipeline del gold v3.

### `archive/dev/` — herramientas de desarrollo
- `_build_nb_*.py` — generadores de los notebooks (los notebooks finales se
  construyeron con estos scripts).
- `_validar_*.py` — validadores que ejecutaban los notebooks en CPU con datos mini
  para detectar errores antes de correr en GPU.

### `archive/logs/` — logs de corridas (transitorios)
Salida de consola de entrenamientos/experimentos previos. Solo referencia.
