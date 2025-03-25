# Treinamento de Modelo YOLO para Análise de Futebol

Este diretório contém todos os arquivos necessários para treinar um modelo YOLOv12 especializado em detecção de jogadores, goleiros, árbitros e bola em partidas de futebol.

## Estrutura do Diretório

```
training/
├── data.yaml            # Configuração do dataset (classes e caminhos)
├── train.py             # Script principal de treinamento
├── yolo12n.pt           # Modelo base YOLOv12n
├── train/               # Imagens e anotações de treinamento
│   ├── images/          # Imagens para treinamento
│   └── labels/          # Anotações em formato YOLO
├── valid/               # Imagens e anotações de validação
│   ├── images/          # Imagens para validação
│   └── labels/          # Anotações em formato YOLO
└── test/                # Imagens e anotações de teste
    ├── images/          # Imagens para teste
    └── labels/          # Anotações em formato YOLO
```

## Recursos Especiais do Treinamento

O script `train.py` implementa técnicas avançadas para melhorar a detecção em jogos de futebol:

1. **Tratamento de Campo Verde**: Implementação de segmentação de campo para foco apenas em jogadores dentro do campo.
2. **Variação de Uniformes**: Augmentação de HSV que modifica cores de uniformes para tornar o modelo invariante a cores específicas.
3. **Adaptação para Câmera Fixa**: Menor augmentação de perspectiva e rotação considerando que a câmera é geralmente fixa.
4. **Remoção de Detecções Fora do Campo**: Identificação automática das bordas do campo e filtragem de objetos.

## Como Usar

### Pré-requisitos

```bash
pip install ultralytics opencv-python numpy albumentations
```

### Executar o Treinamento

Para iniciar o treinamento:

```bash
cd training
python train.py
```

O script:
- Cria um dataset aumentado com técnicas específicas para futebol
- Aplica transformações específicas para uniformes e campo
- Treina o modelo YOLOv12n com hiperparâmetros otimizados
- Salva o modelo na pasta `../runs/train/yolo12n_futebol_robusto/`

### Parâmetros Personalizáveis

Você pode modificar os seguintes parâmetros no arquivo `train.py`:

- `MODELO_BASE`: Modelo base a ser utilizado (padrão: 'yolo12n.pt')
- `OUTPUT_DIR`: Diretório de saída para o modelo treinado
- `EPOCHS`: Número de épocas para treinamento (padrão: 200)
- `IMGSZ`: Tamanho das imagens de entrada (padrão: 1280)
- `BATCH`: Tamanho do lote (padrão: 8)
- `PACIENCIA`: Número de épocas para early stopping (padrão: 50)

## Resultados

Após o treinamento, você encontrará:

- Modelo treinado: `../runs/train/yolo12n_futebol_robusto/weights/best.pt`
- Gráficos de desempenho: `../runs/train/yolo12n_futebol_robusto/*.png`
- Configurações usadas: `../runs/train/yolo12n_futebol_robusto/args.yaml`

Este modelo pode ser utilizado diretamente com os scripts na pasta `../analyzer/` para análise de métricas em vídeos de futebol. 