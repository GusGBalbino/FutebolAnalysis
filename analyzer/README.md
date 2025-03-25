# Análise de Métricas de Futebol com Visão Computacional

Este módulo utiliza modelos YOLO treinados e o rastreador DeepSORT para extrair métricas avançadas de partidas de futebol a partir de vídeos.

## Estrutura do Diretório

```
analyzer/
├── scripts/                         # Scripts de análise
│   ├── football_analyzer_v2.py      # Analisador principal de métricas
│   ├── football_tracker.py          # Implementação do rastreador com DeepSORT e Kalman
│   └── test_tracker.py              # Script para testar o rastreamento visual
├── models/                          # Diretório para modelos treinados
├── resultados/                      # Resultados das análises (relatórios e gráficos)
├── runs/                            # Link para modelos treinados
└── video_test.mp4                   # Vídeo de exemplo para teste
```

## Recursos e Métricas

O analisador de futebol extrai as seguintes métricas:

- **Posse de Bola**: Percentual de tempo que cada time está com a posse da bola
- **Distância Percorrida**: Distância total percorrida por cada time
- **Velocidade**: Velocidade média e máxima por time
- **Mapas de Calor**: Visualização de densidade de posicionamento de cada time e da bola
- **Análise por Zonas**: Tempo de bola em cada terço do campo (defensivo, meio, ofensivo)
- **Trajetórias**: Rastreamento da movimentação de jogadores e bola

## Tecnologias Implementadas

- **YOLOv12**: Detecção de jogadores, goleiros, árbitros e bola
- **DeepSORT**: Rastreamento robusto de múltiplos objetos
- **Filtro de Kalman**: Implementação especializada para rastreamento da bola
- **Calibração de Campo**: Conversão de coordenadas de pixel para metros
- **Projeção Homográfica**: Opção para calibração precisa com homografia 

## Como Usar

### Instalação de Dependências

```bash
pip install ultralytics torch opencv-python deep-sort-realtime matplotlib numpy tqdm
```

### Testar o Rastreamento Visual

Para visualizar o funcionamento do rastreamento em um vídeo:

```bash
cd analyzer
python scripts/test_tracker.py --video video_test.mp4 --output rastreamento.mp4
```

### Analisar um Vídeo Completo

Para executar a análise completa e gerar relatório de métricas:

```bash
cd analyzer
python scripts/football_analyzer_v2.py --video video_test.mp4 --model models/best.pt --output resultados --save_vis
```

### Opções Disponíveis

O script principal `football_analyzer_v2.py` suporta as seguintes opções:

- `--model`: Caminho para o modelo YOLO treinado
- `--video`: Caminho para o vídeo a ser analisado
- `--output`: Diretório para salvar os resultados
- `--save_vis`: Flag para salvar vídeo com visualização do rastreamento
- `--vis_path`: Caminho personalizado para salvar o vídeo com visualização
- `--device`: Dispositivo para inferência ('cpu', 'cuda', 'mps')

## Resultados e Visualizações

Após a análise, o seguinte é gerado no diretório de saída:

1. **Relatório Textual** (`football_metrics_report.txt`):
   - Estatísticas detalhadas de posse de bola
   - Velocidades médias e máximas por equipe
   - Distância total percorrida por equipe
   - Posse de bola por zona do campo

2. **Visualizações**:
   - `possession.png`: Gráfico de pizza da posse de bola
   - `field_heatmaps.png`: Mapas de calor dos times e da bola sobrepostos em campo de futebol
   - `field_zones.png`: Visualização das zonas do campo com percentuais de posse
   - `team_speeds.png`: Comparativo de velocidades entre as equipes
   - `team_distances.png`: Comparativo de distâncias percorridas
   - `zone_distribution.png`: Distribuição da bola pelo campo

3. **Vídeo com Visualização** (opcional):
   - Rastreamento dos jogadores e bola com trajetórias
   - Identificação de times por cores
   - Informações em tempo real sobre posse de bola 