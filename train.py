from ultralytics import YOLO

# Baixar o modelo pre-treinado
model = YOLO('yolo12n.pt')

# Realizar o fine tuning com os dados disponíveis
# O arquivo data.yaml contém as informações sobre os dados de treinamento
results = model.train(
    data='data.yaml',      # Arquivo de configuração dos dados
    epochs=150,            # Mais épocas para melhor convergência
    imgsz=1280,            # Tamanho maior para melhor detecção de objetos pequenos/distantes
    patience=50,           # Early stopping
    batch=8,               # Batch menor para acomodar imagens maiores
    project='runs/train',  # Diretório para salvar os resultados
    name='yolo12n_futbol_metrics',  # Nome do experimento
    optimizer='AdamW',     # Otimizador mais adequado para detalhes finos
    lr0=0.001,             # Taxa de aprendizado inicial
    lrf=0.01,              # Taxa de aprendizado final
    augment=True,          # Usar data augmentation
    mosaic=1.0,            # Usar mosaico para aumentar diversidade
    mixup=0.15,            # Usar mixup para melhorar generalização
    degrees=10.0           # Rotações para generalização de ângulos de câmera
)

# Validar o modelo
results = model.val()

# Salvar o modelo fine-tuned
model.export(format='pt', save_dir='./')

# Código de exemplo para usar o modelo com rastreamento
# (Descomente e use após treinamento)
"""
def process_video(video_path, output_path):
    from ultralytics.trackers import BYTETracker
    
    # Carregue o modelo treinado
    model = YOLO('runs/train/yolo12n_futbol_metrics/weights/best.pt')
    
    # Configure o rastreador
    tracker = BYTETracker(
        track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.8,
        frame_rate=30
    )
    
    # Processe o vídeo
    results = model.track(
        source=video_path,
        tracker=tracker,
        save=True,
        project='runs/track',
        name='football_metrics',
        stream=True
    )
    
    # Aqui você pode adicionar código para calcular métricas
    # como posse de bola, velocidade, distância, etc.
    # baseado nos resultados do rastreamento
"""