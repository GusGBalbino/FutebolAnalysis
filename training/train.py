from ultralytics import YOLO
import torch
import os
import yaml
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
import shutil
import glob
from tqdm import tqdm

# Configurações
MODELO_BASE = 'yolo12n.pt'
OUTPUT_DIR = 'runs/train/yolo12n_futebol_robusto'
EPOCHS = 200
IMGSZ = 1280
BATCH = 8
PACIENCIA = 50

# Criação de diretório temporário para dados aumentados
TEMP_DIR = 'temp_dataset'
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(os.path.join(TEMP_DIR, 'images'), exist_ok=True)
os.makedirs(os.path.join(TEMP_DIR, 'labels'), exist_ok=True)

def normalize_bboxes(bboxes, img_width, img_height):
    """
    Normaliza as coordenadas das bounding boxes para o intervalo [0, 1]
    
    Args:
        bboxes: Lista de bounding boxes no formato [x_min, y_min, x_max, y_max, class_id]
        img_width: Largura da imagem
        img_height: Altura da imagem
        
    Returns:
        Lista de bounding boxes normalizadas
    """
    normalized_bboxes = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max, class_id = bbox
        # Normalizar coordenadas
        x_min = max(0, min(1, x_min / img_width))
        y_min = max(0, min(1, y_min / img_height))
        x_max = max(0, min(1, x_max / img_width))
        y_max = max(0, min(1, y_max / img_height))
        normalized_bboxes.append([x_min, y_min, x_max, y_max, class_id])
    return normalized_bboxes

def preprocess_and_augment():
    """Pré-processamento avançado e augmentação específica para futebol"""
    
    # Carregar o arquivo de configuração original
    with open('data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Diretórios dos dados originais
    train_dir = os.path.abspath('train')
    train_images = os.path.join(train_dir, 'images')
    train_labels = os.path.join(train_dir, 'labels')
    
    # Pipeline de augmentação com foco em problemas específicos de futebol
    transform = A.Compose([
        # Variação de cores - simula diferentes uniformes e condições de iluminação
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.2, p=0.7),
        
        # Variação de HSV com foco em verde - ajuda a lidar com o campo verde
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        
        # Simulação de diferentes condições de iluminação
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        
        # Transformações geométricas moderadas - a câmera é fixa, mas pode haver pequenas variações
        A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-10, 10), p=0.5),
        
        # Adiciona ruído para aumentar robustez
        A.GaussNoise(std_range=[0.1, 0.2], mean_range=[0, 0], p=0.3),
        
        # Blur para simular diferentes qualidades de vídeo
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        
        # Ocasionalmente adiciona sombras para aumentar robustez
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    # Processar imagens de treinamento
    image_files = [os.path.join(train_images, f) for f in os.listdir(train_images) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Processando {len(image_files)} imagens com augmentação...")
    
    for i, img_path in enumerate(image_files):
        # Carregar imagem
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Erro ao carregar imagem: {img_path}")
            continue
            
        h, w, _ = img.shape
        
        # Carregar labels
        label_path = os.path.join(train_labels, img_name.rsplit('.', 1)[0] + '.txt')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            bboxes = []
            class_labels = []
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    # Filtrar objetos muito pequenos (provável ruído ou fora do campo)
                    if width * height > 0.0001:  # Limite mínimo de área
                        # Converter para formato [x_min, y_min, x_max, y_max]
                        x_min = x_center - width/2
                        y_min = y_center - height/2
                        x_max = x_center + width/2
                        y_max = y_center + height/2
                        
                        # Garantir que as coordenadas estejam no intervalo [0, 1]
                        x_min = max(0, min(1, x_min))
                        y_min = max(0, min(1, y_min))
                        x_max = max(0, min(1, x_max))
                        y_max = max(0, min(1, y_max))
                        
                        # Converter de volta para formato YOLO
                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        width = x_max - x_min
                        height = y_max - y_min
                        
                        bboxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)
            
            # Aplicar augmentação se houver bounding boxes
            if bboxes:
                try:
                    # Augmentação normal
                    transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)
                    aug_img = transformed['image']
                    aug_bboxes = transformed['bboxes']
                    aug_labels = transformed['class_labels']
                    
                    # Salvar imagem augmentada
                    aug_img_path = os.path.join(TEMP_DIR, 'images', f"aug_{i}_{img_name}")
                    cv2.imwrite(aug_img_path, aug_img)
                    
                    # Salvar labels augmentados
                    aug_label_path = os.path.join(TEMP_DIR, 'labels', f"aug_{i}_{img_name.rsplit('.', 1)[0]}.txt")
                    with open(aug_label_path, 'w') as f:
                        for label_idx, bbox in enumerate(aug_bboxes):
                            x_center, y_center, width, height = bbox
                            class_id = aug_labels[label_idx]
                            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                except Exception as e:
                    print(f"Erro ao processar imagem {img_path}: {str(e)}")
                    continue
    
    # Criar novo arquivo de configuração para dataset aumentado
    temp_data_yaml = os.path.join(TEMP_DIR, 'data.yaml')
    
    # Atualizar caminhos para usar diretórios relativos ao script atual
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_config['train'] = os.path.abspath(os.path.join(TEMP_DIR, 'images'))
    data_config['val'] = os.path.abspath(os.path.join(script_dir, 'valid/images'))
    data_config['test'] = os.path.abspath(os.path.join(script_dir, 'test/images'))
    
    with open(temp_data_yaml, 'w') as f:
        yaml.dump(data_config, f)
    
    print(f"Augmentação completa. Dataset temporário criado em {TEMP_DIR}")
    return temp_data_yaml

# Aplicar pré-processamento e augmentação
augmented_data_yaml = preprocess_and_augment()

# Carregar o modelo base
print(f"Carregando modelo base: {MODELO_BASE}")
model = YOLO(MODELO_BASE)

# Configurar hiperparâmetros avançados para o treinamento
print("Iniciando treinamento com augmentação e hiperparâmetros otimizados")
results = model.train(
    data=augmented_data_yaml,      # Arquivo de configuração dos dados aumentados
    epochs=EPOCHS,                 # Mais épocas para melhor convergência
    imgsz=IMGSZ,                   # Tamanho maior para melhor detecção de objetos pequenos/distantes
    patience=PACIENCIA,            # Early stopping
    batch=BATCH,                   # Batch size
    project='runs/train',          # Diretório para salvar os resultados
    name='yolo12n_futebol_robusto',  # Nome do experimento
    optimizer='AdamW',             # Otimizador mais adequado para detalhes finos
    lr0=0.001,                     # Taxa de aprendizado inicial
    lrf=0.01,                      # Taxa de aprendizado final
    augment=True,                  # Usar augmentação integrada do YOLO
    mosaic=1.0,                    # Usar mosaico para aumentar diversidade
    mixup=0.15,                    # Usar mixup para melhorar generalização
    degrees=10.0,                  # Rotações para generalização de ângulos de câmera
    hsv_h=0.2,                     # Augmentação de matiz (cor)
    hsv_s=0.7,                     # Augmentação de saturação maior para lidar com campo verde
    hsv_v=0.4,                     # Augmentação de brilho
    perspective=0.001,             # Menor perspectiva (câmera fixa)
    # Parâmetros avançados de treinamento
    weight_decay=0.0005,           # Regularização para evitar overfitting
    cos_lr=True,                   # Agendamento de taxa de aprendizado por cosseno
    warmup_epochs=3.0,             # Aquecimento para estabilidade inicial
    box=7.5,                       # Maior peso para perda de box
    cls=0.3,                       # Peso para perda de classificação
    dfl=1.5,                       # Peso para distribution focal loss
)

# Validar o modelo
print("Validando modelo treinado...")
results = model.val()

# Exportar o modelo optimizado
print("Exportando modelo treinado...")
model.export(format='pt', save_dir=OUTPUT_DIR)

print(f"Treinamento concluído! Modelo salvo em {OUTPUT_DIR}")

# Limpar arquivos temporários
if os.path.exists(TEMP_DIR):
    print("Limpando arquivos temporários...")
    shutil.rmtree(TEMP_DIR)

print("Processo finalizado com sucesso!")
