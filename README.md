# Sistema de Análise de Partidas de Futebol com IA

Este projeto implementa um sistema completo para análise de métricas em partidas de futebol usando visão computacional e inteligência artificial. O sistema é dividido em dois módulos principais: treinamento de modelos e análise de métricas.

## Estrutura do Projeto

```
.
├── training/                # Módulo de treinamento de modelos
│   ├── data.yaml            # Configuração de dataset
│   ├── train.py             # Script de treinamento
│   ├── yolo12n.pt           # Modelo base YOLOv12n
│   ├── train/               # Dados de treinamento
│   ├── valid/               # Dados de validação
│   ├── test/                # Dados de teste 
│   └── README.md            # Documentação específica de treinamento
│
├── analyzer/                # Módulo de análise de métricas
│   ├── scripts/             # Scripts de análise
│   │   ├── football_analyzer_v2.py
│   │   ├── football_tracker.py
│   │   └── test_tracker.py
│   ├── models/              # Diretório para modelos
│   ├── resultados/          # Resultados das análises
│   ├── video_test.mp4       # Vídeo de exemplo
│   └── README.md            # Documentação específica de análise
│
├── runs/                    # Resultados de treinamentos (modelos)
│   └── train/
│       └── yolo12n_futbol_metrics16/
└── README.md                # Este arquivo
```

## Visão Geral

Este sistema permite:

1. **Treinar modelos YOLO** especializados em detecção de objetos em partidas de futebol (jogadores, goleiros, árbitro e bola)
2. **Analisar métricas avançadas** a partir de vídeos de partidas, incluindo:
   - Posse de bola por equipe
   - Velocidade e distância percorrida pelos jogadores
   - Mapas de calor de posicionamento
   - Distribuição da bola pelo campo

## Como Usar

### Treinamento de Modelos

Para treinar modelos customizados em seus próprios dados:

```bash
cd training
python train.py
```

Consulte [training/README.md](training/README.md) para instruções detalhadas sobre o processo de treinamento.

### Análise de Métricas em Vídeos

Para analisar métricas em vídeos de partidas:

```bash
cd analyzer
python scripts/football_analyzer_v2.py --video video_test.mp4 --model models/best.pt --output resultados --save_vis
```

Consulte [analyzer/README.md](analyzer/README.md) para todas as opções disponíveis e exemplos de uso.

## Características Especiais

- **Rastreamento Aprimorado**: Filtro de Kalman especializado para melhor rastreamento da bola
- **Invariância a Uniformes**: Técnicas de data augmentation para lidar com diferentes cores de uniforme
- **Segmentação de Campo**: Detecção automática do campo para filtrar detecções externas
- **Calibração Métrica**: Conversão precisa de pixels para metros baseada no tamanho do campo

## Pré-requisitos

- Python 3.8+
- PyTorch
- Ultralytics YOLO
- OpenCV
- DeepSORT
- Albumentations (para treinamento)
- Matplotlib (para visualizações)
- NumPy, tqdm

## Fluxo de Trabalho Recomendado

1. Treine ou utilize um modelo pré-treinado do diretório `runs/train/`
2. Teste o rastreamento visualmente com `test_tracker.py`
3. Execute a análise completa com `football_analyzer_v2.py`
4. Examine os relatórios e visualizações gerados no diretório de saída 