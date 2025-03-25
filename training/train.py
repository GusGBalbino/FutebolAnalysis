from ultralytics import YOLO
import torch
import os
import yaml
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
import shutil

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
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.5),
        
        # Adiciona ruído para aumentar robustez
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        
        # Blur para simular diferentes qualidades de vídeo
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        
        # Ocasionalmente adiciona sombras para aumentar robustez
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=3, p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    # Processar imagens de treinamento
    image_files = [os.path.join(train_images, f) for f in os.listdir(train_images) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Processando {len(image_files)} imagens com augmentação...")
    
    for i, img_path in enumerate(image_files):
        # Carregar imagem
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
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
                        bboxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)
            
            # Aplicar augmentação se houver bounding boxes
            if bboxes:
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
                
                # Criar versão com mudança de cor nos uniformes
                # Isso ajuda o modelo a ser invariante às cores específicas
                hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                
                # Mudar matiz (cor) mantendo saturação e valor
                # Alterando cores dos uniformes sem afetar o verde do campo
                hue_shift = np.random.randint(20, 160)
                hsv_img[..., 0] = (hsv_img[..., 0] + hue_shift) % 180
                
                # Converter de volta para BGR
                color_varied_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
                
                # Salvar imagem com cores variadas
                color_img_path = os.path.join(TEMP_DIR, 'images', f"color_{i}_{img_name}")
                cv2.imwrite(color_img_path, color_varied_img)
                
                # Salvar labels (mesmas coordenadas)
                color_label_path = os.path.join(TEMP_DIR, 'labels', f"color_{i}_{img_name.rsplit('.', 1)[0]}.txt")
                with open(color_label_path, 'w') as f:
                    for label_idx, bbox in enumerate(bboxes):
                        x_center, y_center, width, height = bbox
                        class_id = class_labels[label_idx]
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                
                # Técnica adicional: Aplicar máscara somente ao campo
                # Isso ensina o modelo a focar apenas nos objetos dentro do campo
                if i % 4 == 0:  # Aplicar em 25% das imagens
                    # Detector simples de campo baseado em cor verde
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    lower_green = np.array([35, 40, 40])
                    upper_green = np.array([90, 255, 255])
                    mask = cv2.inRange(hsv, lower_green, upper_green)
                    
                    # Dilatação para incluir linhas brancas
                    kernel = np.ones((5, 5), np.uint8)
                    mask = cv2.dilate(mask, kernel, iterations=2)
                    
                    # Encontrar contorno do campo
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # Pegar o maior contorno (provavelmente o campo)
                        field_contour = max(contours, key=cv2.contourArea)
                        
                        # Criar nova máscara apenas com contorno do campo
                        field_mask = np.zeros_like(mask)
                        cv2.drawContours(field_mask, [field_contour], 0, 255, -1)
                        
                        # Aplicar máscara nas labels - remover objetos fora do campo
                        filtered_bboxes = []
                        filtered_labels = []
                        
                        for label_idx, bbox in enumerate(bboxes):
                            x_center, y_center, width, height = bbox
                            
                            # Converter para coordenadas de pixel
                            x_px = int(x_center * w)
                            y_px = int(y_center * h)
                            
                            # Verificar se o centro do objeto está dentro do campo
                            if field_mask[y_px, x_px] > 0:
                                filtered_bboxes.append(bbox)
                                filtered_labels.append(class_labels[label_idx])
                        
                        # Salvar imagem com máscara de campo
                        field_img_path = os.path.join(TEMP_DIR, 'images', f"field_{i}_{img_name}")
                        cv2.imwrite(field_img_path, img)
                        
                        # Salvar apenas as labels dentro do campo
                        field_label_path = os.path.join(TEMP_DIR, 'labels', f"field_{i}_{img_name.rsplit('.', 1)[0]}.txt")
                        with open(field_label_path, 'w') as f:
                            for label_idx, bbox in enumerate(filtered_bboxes):
                                x_center, y_center, width, height = bbox
                                class_id = filtered_labels[label_idx]
                                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    
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
