import json
import re

nb_path = 'd:/Tesis/ABSA_Turismo_Fase2_V3/notebooks/02_absa_bert_textcnn_inferencia_matriz_v3_completo.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Modify Hyperparams
        if 'LEARNING_RATE = 1e-5' in source:
            source = source.replace('LEARNING_RATE = 1e-5', 'LEARNING_RATE = 5e-6')
            source = source.replace('WEIGHT_DECAY = 0.10', 'WEIGHT_DECAY = 0.30')
            source = source.replace('DROPOUT = 0.50', 'DROPOUT = 0.65')
            source = source.replace('EPOCHS = 4', 'EPOCHS = 6')
            
        # Modify Class Weights
        if 'def compute_class_weights' in source:
            old_weight_code = 'weights.append(total / (len(VALID_LABELS) * count))'
            new_weight_code = '''w = total / (len(VALID_LABELS) * count)
            if label == "neutro":
                w *= 2.0
            weights.append(w)'''
            if old_weight_code in source:
                source = source.replace(old_weight_code, new_weight_code)
                
        # Update source in cell
        if ''.join(cell['source']) != source:
            lines = [line + '\n' for line in source.split('\n')]
            # Remove trailing newline from the very last element to match original format
            if lines:
                lines[-1] = lines[-1].rstrip('\n')
            cell['source'] = lines

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Notebook updated successfully.")
