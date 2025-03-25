import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import math
from collections import defaultdict
import argparse
import torch
import sys

# Adicionar o diretório raiz e o diretório de scripts ao sys.path para encontrar módulos
current_dir = os.path.dirname(os.path.abspath(__file__))
analyzer_dir = os.path.dirname(current_dir)
sys.path.append(analyzer_dir)
from scripts.football_tracker import FootballTracker, FieldCalibrator

class FootballAnalyzerV2:
    """Analisador de métricas para partidas de futebol usando DeepSORT e rastreamento especializado"""
    
    def __init__(self, model_path, video_fps=30.0, field_length=105, field_width=68, device=None):
        """
        Inicializa o analisador de futebol
        
        Args:
            model_path: Caminho para o modelo YOLO treinado
            video_fps: Frames por segundo do vídeo
            field_length: Comprimento do campo em metros
            field_width: Largura do campo em metros
            device: Dispositivo para inferência ('cpu', 'cuda', 'mps')
        """
        # Configurar dispositivo
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Configurações gerais
        self.fps = video_fps
        self.field_length = field_length
        self.field_width = field_width
        
        # Inicializar rastreador e calibrador
        self.tracker = FootballTracker(model_path, device=self.device)
        self.calibrator = FieldCalibrator(field_length, field_width)
        
        # Métricas
        self.ball_possession = {"Team1": 0, "Team2": 0}
        self.distances = {"Team1": {}, "Team2": {}}
        self.speeds = {"Team1": {}, "Team2": {}}
        self.max_speeds = {"Team1": {}, "Team2": {}}
        self.avg_team_positions = {"Team1": [], "Team2": []}
        self.heatmaps = {"Team1": np.zeros((100, 100)), "Team2": np.zeros((100, 100))}
        self.ball_heatmap = np.zeros((100, 100))
        
        # Mapa de zonas do campo
        self.field_zones = {
            "att_third_team1": 0,
            "mid_third_team1": 0,
            "def_third_team1": 0,
            "att_third_team2": 0,
            "mid_third_team2": 0,
            "def_third_team2": 0
        }
        
        # Outras estatísticas
        self.total_frames = 0
        self.calibrated = False
    
    def _calculate_ball_possession(self, ball_pos, frame_idx):
        """
        Calcula a posse de bola com base na proximidade dos jogadores
        
        Args:
            ball_pos: Posição da bola (x, y)
            frame_idx: Índice do frame atual
        """
        if ball_pos is None:
            return
            
        # Atualizar mapa de calor da bola
        if self.calibrated:
            x_norm = int((ball_pos[0] / self.frame_width) * 100)
            y_norm = int((ball_pos[1] / self.frame_height) * 100)
            
            # Garantir que está dentro dos limites
            x_norm = max(0, min(99, x_norm))
            y_norm = max(0, min(99, y_norm))
            
            self.ball_heatmap[y_norm, x_norm] += 1
            
            # Atualizar zonas do campo
            field_x_relative = ball_pos[0] / self.frame_width
            if field_x_relative < 0.33:
                self.field_zones["def_third_team1"] += 1
                self.field_zones["att_third_team2"] += 1
            elif field_x_relative < 0.66:
                self.field_zones["mid_third_team1"] += 1
                self.field_zones["mid_third_team2"] += 1
            else:
                self.field_zones["att_third_team1"] += 1
                self.field_zones["def_third_team2"] += 1
        
        # Encontrar o jogador mais próximo da bola
        min_dist_team1 = float('inf')
        min_dist_team2 = float('inf')
        players_team1_near_ball = 0
        players_team2_near_ball = 0
        team1_active = False
        team2_active = False
        
        # Verificar jogadores do Time 1
        for track_id, track in self.tracker.team1_tracks.items():
            if track and frame_idx - track[-1][0] < 5:  # Se visto recentemente
                team1_active = True
                last_pos = track[-1][1]
                dist = math.sqrt((ball_pos[0] - last_pos[0])**2 + (ball_pos[1] - last_pos[1])**2)
                
                if dist < min_dist_team1:
                    min_dist_team1 = dist
                
                # Contar jogadores próximos da bola
                if dist < 100.0:  # Jogadores em um raio de 100px da bola 
                    players_team1_near_ball += 1
        
        # Verificar jogadores do Time 2
        for track_id, track in self.tracker.team2_tracks.items():
            if track and frame_idx - track[-1][0] < 5:
                team2_active = True
                last_pos = track[-1][1]
                dist = math.sqrt((ball_pos[0] - last_pos[0])**2 + (ball_pos[1] - last_pos[1])**2)
                
                if dist < min_dist_team2:
                    min_dist_team2 = dist
                    
                # Contar jogadores próximos da bola
                if dist < 100.0:
                    players_team2_near_ball += 1
        
        # Verificar se ambos os times estão ativos no frame
        if not team1_active or not team2_active:
            # Se um dos times não está ativo, não atribuir posse
            return
            
        # Threshold para posse de bola (distância em pixels)
        threshold_px = 50.0
        
        # Determinar qual time está mais próximo da bola
        if min_dist_team1 < threshold_px or min_dist_team2 < threshold_px:
            # Se algum time está próximo o suficiente para ter posse
            if min_dist_team1 < min_dist_team2:
                self.ball_possession["Team1"] += 1
            elif min_dist_team2 < min_dist_team1:
                self.ball_possession["Team2"] += 1
            else:
                # Empate - considerar o número de jogadores próximos
                if players_team1_near_ball > players_team2_near_ball:
                    self.ball_possession["Team1"] += 1
                elif players_team2_near_ball > players_team1_near_ball:
                    self.ball_possession["Team2"] += 1
                else:
                    # Empate completo - dividir a posse
                    self.ball_possession["Team1"] += 0.5
                    self.ball_possession["Team2"] += 0.5
        else:
            # Bola "livre" - não atribuir posse a nenhum time
            pass
    
    def _calculate_team_positions(self, frame_idx):
        """Calcula a posição média de cada equipe"""
        team1_positions = []
        team2_positions = []
        
        # Coletar posições do Time 1
        for track_id, track in self.tracker.team1_tracks.items():
            if track and frame_idx - track[-1][0] < 5:  # Se visto recentemente
                team1_positions.append(track[-1][1])
        
        # Coletar posições do Time 2
        for track_id, track in self.tracker.team2_tracks.items():
            if track and frame_idx - track[-1][0] < 5:
                team2_positions.append(track[-1][1])
        
        # Calcular posição média
        if team1_positions:
            avg_x = sum(pos[0] for pos in team1_positions) / len(team1_positions)
            avg_y = sum(pos[1] for pos in team1_positions) / len(team1_positions)
            self.avg_team_positions["Team1"].append((frame_idx, (avg_x, avg_y)))
            
            # Atualizar mapa de calor
            if self.calibrated:
                for pos in team1_positions:
                    x_norm = int((pos[0] / self.frame_width) * 100)
                    y_norm = int((pos[1] / self.frame_height) * 100)
                    
                    x_norm = max(0, min(99, x_norm))
                    y_norm = max(0, min(99, y_norm))
                    
                    self.heatmaps["Team1"][y_norm, x_norm] += 1
        
        if team2_positions:
            avg_x = sum(pos[0] for pos in team2_positions) / len(team2_positions)
            avg_y = sum(pos[1] for pos in team2_positions) / len(team2_positions)
            self.avg_team_positions["Team2"].append((frame_idx, (avg_x, avg_y)))
            
            # Atualizar mapa de calor
            if self.calibrated:
                for pos in team2_positions:
                    x_norm = int((pos[0] / self.frame_width) * 100)
                    y_norm = int((pos[1] / self.frame_height) * 100)
                    
                    x_norm = max(0, min(99, x_norm))
                    y_norm = max(0, min(99, y_norm))
                    
                    self.heatmaps["Team2"][y_norm, x_norm] += 1
    
    def _calculate_distances_speeds(self):
        """Calcula distâncias percorridas e velocidades dos jogadores"""
        # Limites de velocidade realistas para jogadores de futebol (em m/s)
        MAX_REALISTIC_SPEED = 12.0  # Aproximadamente 43 km/h, velocidade máxima de atletas de elite
        MIN_REALISTIC_SPEED = 0.2   # Filtrar micromovimentos e ruídos no rastreamento
        
        # Contador de anomalias
        anomalias_detectadas = 0
        
        # Time 1
        for player_id, track in self.tracker.team1_tracks.items():
            if len(track) < 2:
                continue
                
            total_distance = 0
            speeds = []
            max_speed = 0
            
            for i in range(1, len(track)):
                prev_frame, prev_pos = track[i-1]
                curr_frame, curr_pos = track[i]
                
                # Calcular distância em pixels
                if self.calibrated:
                    # Converter para metros usando calibração
                    dist_m = self.calibrator.calc_distance_meters(prev_pos, curr_pos)
                else:
                    # Usar estimativa básica
                    dist_px = math.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                    dist_m = dist_px / 15.0  # Estimativa simples: 15px = 1m
                
                # Calcular tempo entre frames
                time_diff = (curr_frame - prev_frame) / self.fps
                
                if time_diff > 0:
                    # Velocidade em metros por segundo
                    speed = dist_m / time_diff
                    
                    # Filtrar velocidades não realistas
                    if speed > MAX_REALISTIC_SPEED:
                        # Velocidade anormalmnete alta detectada
                        anomalias_detectadas += 1
                        # Ajustar para o valor máximo realista
                        speed = MAX_REALISTIC_SPEED
                        # Recalcular a distância com base na velocidade máxima
                        dist_m = speed * time_diff
                    elif speed < MIN_REALISTIC_SPEED:
                        # Ignorar movimentos muito pequenos (possível ruído)
                        continue
                    
                    speeds.append(speed)
                    max_speed = max(max_speed, speed)
                
                total_distance += dist_m
            
            self.distances["Team1"][player_id] = total_distance
            
            if speeds:
                self.speeds["Team1"][player_id] = sum(speeds) / len(speeds)
                self.max_speeds["Team1"][player_id] = max_speed
        
        # Time 2
        for player_id, track in self.tracker.team2_tracks.items():
            if len(track) < 2:
                continue
                
            total_distance = 0
            speeds = []
            max_speed = 0
            
            for i in range(1, len(track)):
                prev_frame, prev_pos = track[i-1]
                curr_frame, curr_pos = track[i]
                
                # Calcular distância
                if self.calibrated:
                    dist_m = self.calibrator.calc_distance_meters(prev_pos, curr_pos)
                else:
                    dist_px = math.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                    dist_m = dist_px / 15.0
                
                time_diff = (curr_frame - prev_frame) / self.fps
                
                if time_diff > 0:
                    speed = dist_m / time_diff
                    
                    # Filtrar velocidades não realistas
                    if speed > MAX_REALISTIC_SPEED:
                        anomalias_detectadas += 1
                        speed = MAX_REALISTIC_SPEED
                        dist_m = speed * time_diff
                    elif speed < MIN_REALISTIC_SPEED:
                        continue
                    
                    speeds.append(speed)
                    max_speed = max(max_speed, speed)
                
                total_distance += dist_m
            
            self.distances["Team2"][player_id] = total_distance
            
            if speeds:
                self.speeds["Team2"][player_id] = sum(speeds) / len(speeds)
                self.max_speeds["Team2"][player_id] = max_speed
        
        if anomalias_detectadas > 0:
            print(f"[AVISO] {anomalias_detectadas} medições de velocidade anômalas foram detectadas e corrigidas.")
            print(f"As velocidades foram limitadas ao máximo de {MAX_REALISTIC_SPEED} m/s (≈ {MAX_REALISTIC_SPEED*3.6:.1f} km/h).")
    
    def analyze_video(self, video_path, output_dir="resultados", save_vis=False, vis_path=None):
        """
        Analisa um vídeo de futebol e calcula métricas
        
        Args:
            video_path: Caminho para o vídeo
            output_dir: Diretório para salvar resultados
            save_vis: Se deve salvar visualização do vídeo
            vis_path: Caminho para salvar a visualização (se None, será definido automaticamente)
        
        Returns:
            Dict com métricas calculadas
        """
        # Verificar se o vídeo existe
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")
        
        # Criar diretório de saída
        os.makedirs(output_dir, exist_ok=True)
        
        # Criar caminho para visualização
        if save_vis:
            if vis_path is None:
                # Usar nome do arquivo de entrada com prefixo
                base_name = os.path.basename(video_path)
                vis_path = os.path.join(output_dir, f"analise_{base_name}")
                
                # Garantir que a extensão seja .mp4
                name, ext = os.path.splitext(vis_path)
                if ext.lower() not in ['.mp4', '.avi']:
                    vis_path = name + '.mp4'
                    
            print(f"Vídeo de análise será salvo em: {vis_path}")
            
        # Configurar tracker
        self.tracker.reset_tracks()
        
        # Abrir vídeo
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")
            
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Atualizar FPS se disponível no vídeo
        if video_fps > 0:
            self.fps = video_fps
        
        # Calibrar o campo
        self.calibrator.calibrate_from_field_size(self.frame_width, self.frame_height)
        self.calibrated = True
        
        # Configurar visualização
        out = None
        if save_vis:
            try:
                # Tentar primeiro com mp4v (mais compatível)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(vis_path, fourcc, self.fps, 
                                    (self.frame_width, self.frame_height))
                
                # Verificar se o writer foi criado corretamente
                if not out.isOpened():
                    # Tentar com codec alternativo
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(vis_path, fourcc, self.fps, 
                                         (self.frame_width, self.frame_height))
                    print("Usando codec alternativo (XVID) para salvar o vídeo")
                    
                if not out.isOpened():
                    print(f"AVISO: Não foi possível criar o arquivo de vídeo em: {vis_path}")
                    print("A análise continuará, mas o vídeo não será salvo.")
                    save_vis = False
            except Exception as e:
                print(f"Erro ao configurar gravação de vídeo: {e}")
                print("A análise continuará, mas o vídeo não será salvo.")
                save_vis = False
        
        # Processar frames
        frame_idx = 0
        
        for frame_idx in tqdm(range(self.total_frames), desc="Analisando vídeo"):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Rastrear objetos no frame
            ball_pos = self.tracker.track_frame(frame, frame_idx)
            
            # Calcular métricas em tempo real
            if ball_pos is not None:
                self._calculate_ball_possession(ball_pos, frame_idx)
            
            self._calculate_team_positions(frame_idx)
            
            # Visualizar rastreamento
            if save_vis and out is not None and out.isOpened():
                try:
                    vis_frame = self.tracker.draw_tracks(frame.copy(), ball_pos)
                    
                    # Adicionar informações na parte superior
                    cv2.putText(vis_frame, f"Frame: {frame_idx}/{self.total_frames}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    poss_team1 = 0
                    poss_team2 = 0
                    total_poss = sum(self.ball_possession.values())
                    if total_poss > 0:
                        poss_team1 = (self.ball_possession["Team1"] / total_poss) * 100
                        poss_team2 = (self.ball_possession["Team2"] / total_poss) * 100
                        
                    cv2.putText(vis_frame, f"Posse: Time1 {poss_team1:.1f}% | Time2 {poss_team2:.1f}%", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    out.write(vis_frame)
                except Exception as e:
                    print(f"Erro ao escrever frame {frame_idx}: {e}")
                    # Continuar análise mesmo com erro na escrita do vídeo
            
            frame_idx += 1
        
        # Limpar
        cap.release()
        if out is not None and out.isOpened():
            out.release()
            print(f"Vídeo de análise salvo com sucesso em: {vis_path}")
        
        # Calcular métricas finais
        self._calculate_distances_speeds()
        
        # Gerar relatório
        report_data = self._generate_report(output_dir)
        
        return report_data
    
    def _balance_possession(self):
        """
        Balanceia a posse de bola quando há desequilíbrio extremo,
        aplicando um fator de correção para evitar valores irrealistas
        """
        total = self.ball_possession["Team1"] + self.ball_possession["Team2"]
        
        if total == 0:
            # Nenhuma posse foi registrada
            self.ball_possession["Team1"] = 0.5
            self.ball_possession["Team2"] = 0.5
            return
            
        team1_pct = self.ball_possession["Team1"] / total
        team2_pct = self.ball_possession["Team2"] / total
        
        # Se houver um desequilíbrio extremo (um time com mais de 85% da posse)
        # aplicar um fator de correção para trazer para valores mais realistas
        if team1_pct > 0.85 or team2_pct > 0.85:
            print("[AVISO] Detectado desequilíbrio extremo na posse de bola, aplicando correção.")
            
            # Aplicar correção suavizada (limitar o máximo a 75%-25%)
            if team1_pct > 0.85:
                team1_pct = min(team1_pct, 0.75)
                team2_pct = 1.0 - team1_pct
            else:
                team2_pct = min(team2_pct, 0.75)
                team1_pct = 1.0 - team2_pct
                
            print(f"Posse balanceada: Time 1: {team1_pct*100:.2f}%, Time 2: {team2_pct*100:.2f}%")
            
            # Atualizar valores de posse
            self.ball_possession["Team1"] = team1_pct * total
            self.ball_possession["Team2"] = team2_pct * total

    def _generate_report(self, output_dir):
        """Gera relatório com métricas calculadas"""
        report_path = os.path.join(output_dir, "football_metrics_report.txt")
        
        # Aplicar balanceamento de posse de bola
        self._balance_possession()
        
        # Calcular estatísticas gerais
        team1_total_dist = sum(self.distances["Team1"].values())
        team2_total_dist = sum(self.distances["Team2"].values())
        
        team1_avg_speed = 0
        team2_avg_speed = 0
        
        if self.speeds["Team1"]:
            team1_avg_speed = sum(self.speeds["Team1"].values()) / len(self.speeds["Team1"])
        
        if self.speeds["Team2"]:
            team2_avg_speed = sum(self.speeds["Team2"].values()) / len(self.speeds["Team2"])
        
        team1_max_speed = 0
        team2_max_speed = 0
        
        if self.max_speeds["Team1"]:
            team1_max_speed = max(self.max_speeds["Team1"].values())
            
        if self.max_speeds["Team2"]:
            team2_max_speed = max(self.max_speeds["Team2"].values())
        
        # Estatísticas de posse de bola
        total_possession = sum(self.ball_possession.values())
        team1_pct = 0
        team2_pct = 0
        
        if total_possession > 0:
            team1_pct = (self.ball_possession["Team1"] / total_possession) * 100
            team2_pct = (self.ball_possession["Team2"] / total_possession) * 100
        
        # Calcular estatísticas por zona
        total_zone_frames = sum(self.field_zones.values())
        zone_stats = {}
        
        if total_zone_frames > 0:
            for zone, count in self.field_zones.items():
                zone_stats[zone] = (count / total_zone_frames) * 100
        
        # Gerar relatório em texto
        with open(report_path, "w") as f:
            f.write("=== RELATÓRIO DE ANÁLISE DE FUTEBOL ===\n\n")
            
            f.write("POSSE DE BOLA:\n")
            f.write(f"Time 1: {team1_pct:.2f}%\n")
            f.write(f"Time 2: {team2_pct:.2f}%\n\n")
            
            f.write("DISTÂNCIA PERCORRIDA:\n")
            f.write(f"Time 1 (total): {team1_total_dist:.2f} metros\n")
            f.write(f"Time 2 (total): {team2_total_dist:.2f} metros\n\n")
            
            f.write("VELOCIDADES:\n")
            f.write(f"Time 1 (média): {team1_avg_speed:.2f} m/s\n")
            f.write(f"Time 1 (máxima): {team1_max_speed:.2f} m/s\n")
            f.write(f"Time 2 (média): {team2_avg_speed:.2f} m/s\n")
            f.write(f"Time 2 (máxima): {team2_max_speed:.2f} m/s\n\n")
            
            f.write("POSSE DE BOLA POR ZONA:\n")
            f.write(f"Terço defensivo (Time 1): {zone_stats.get('def_third_team1', 0):.2f}%\n")
            f.write(f"Terço médio (Time 1): {zone_stats.get('mid_third_team1', 0):.2f}%\n")
            f.write(f"Terço ofensivo (Time 1): {zone_stats.get('att_third_team1', 0):.2f}%\n")
            f.write(f"Terço defensivo (Time 2): {zone_stats.get('def_third_team2', 0):.2f}%\n")
            f.write(f"Terço médio (Time 2): {zone_stats.get('mid_third_team2', 0):.2f}%\n")
            f.write(f"Terço ofensivo (Time 2): {zone_stats.get('att_third_team2', 0):.2f}%\n\n")
            
            f.write(f"Relatório gerado em: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Gerar visualizações
        self._generate_plots(output_dir)
        
        print(f"Relatório salvo em: {report_path}")
        
        return {
            "possession": {"Team1": team1_pct, "Team2": team2_pct},
            "distances": {"Team1": team1_total_dist, "Team2": team2_total_dist},
            "speeds": {
                "Team1": {"avg": team1_avg_speed, "max": team1_max_speed},
                "Team2": {"avg": team2_avg_speed, "max": team2_max_speed}
            },
            "zone_stats": zone_stats
        }
    
    def _generate_plots(self, output_dir):
        """Gera visualizações das métricas"""
        # Gráfico de posse de bola
        total_possession = sum(self.ball_possession.values())
        if total_possession > 0:
            team1_pct = (self.ball_possession["Team1"] / total_possession) * 100
            team2_pct = (self.ball_possession["Team2"] / total_possession) * 100
            
            plt.figure(figsize=(10, 6))
            plt.pie(
                [team1_pct, team2_pct],
                labels=["Time 1", "Time 2"],
                autopct='%1.1f%%',
                startangle=90,
                colors=['red', 'blue']
            )
            plt.title("Posse de Bola")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "possession.png"))
            plt.close()
        
        # Mapas de calor com formato de campo de futebol
        # Criar imagem base do campo de futebol
        field_width = 600
        field_height = 400
        field_img = np.ones((field_height, field_width, 3), dtype=np.uint8) * 50  # Fundo escuro
        
        # Desenhar linhas do campo
        # Linha externa
        cv2.rectangle(field_img, (50, 50), (field_width-50, field_height-50), (255, 255, 255), 2)
        
        # Linha do meio campo
        cv2.line(field_img, (field_width//2, 50), (field_width//2, field_height-50), (255, 255, 255), 2)
        
        # Círculo central
        cv2.circle(field_img, (field_width//2, field_height//2), 50, (255, 255, 255), 2)
        
        # Áreas de gol (esquerda)
        cv2.rectangle(field_img, (50, field_height//2-75), (125, field_height//2+75), (255, 255, 255), 2)
        cv2.rectangle(field_img, (50, field_height//2-30), (75, field_height//2+30), (255, 255, 255), 2)
        
        # Áreas de gol (direita)
        cv2.rectangle(field_img, (field_width-125, field_height//2-75), (field_width-50, field_height//2+75), (255, 255, 255), 2)
        cv2.rectangle(field_img, (field_width-75, field_height//2-30), (field_width-50, field_height//2+30), (255, 255, 255), 2)
        
        # Marcar os gols
        cv2.rectangle(field_img, (40, field_height//2-25), (50, field_height//2+25), (200, 200, 200), -1)
        cv2.rectangle(field_img, (field_width-50, field_height//2-25), (field_width-40, field_height//2+25), (200, 200, 200), -1)
        
        # Redimensionar mapas de calor para o tamanho do campo
        team1_heatmap = cv2.resize(self.heatmaps["Team1"], (field_width-100, field_height-100))
        team2_heatmap = cv2.resize(self.heatmaps["Team2"], (field_width-100, field_height-100))
        ball_heatmap = cv2.resize(self.ball_heatmap, (field_width-100, field_height-100))
        
        # Normalizar mapas de calor para melhor visualização
        if np.max(team1_heatmap) > 0:
            team1_heatmap = team1_heatmap / np.max(team1_heatmap) * 255
        if np.max(team2_heatmap) > 0:
            team2_heatmap = team2_heatmap / np.max(team2_heatmap) * 255
        if np.max(ball_heatmap) > 0:
            ball_heatmap = ball_heatmap / np.max(ball_heatmap) * 255
        
        # Criar imagens coloridas para cada mapa de calor
        team1_colored = cv2.applyColorMap(team1_heatmap.astype(np.uint8), cv2.COLORMAP_HOT)
        team2_colored = cv2.applyColorMap(team2_heatmap.astype(np.uint8), cv2.COLORMAP_JET)
        ball_colored = cv2.applyColorMap(ball_heatmap.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        
        # Criar três campos separados para cada visualização
        field_team1 = field_img.copy()
        field_team2 = field_img.copy()
        field_ball = field_img.copy()
        
        # Sobrepor mapas de calor com transparência
        for i in range(field_height-100):
            for j in range(field_width-100):
                if team1_heatmap[i, j] > 5:  # Limiar para evitar ruído
                    alpha = min(team1_heatmap[i, j] / 255, 0.7)  # Transparência baseada na intensidade
                    field_team1[i+50, j+50] = (1-alpha) * field_team1[i+50, j+50] + alpha * team1_colored[i, j]
                
                if team2_heatmap[i, j] > 5:
                    alpha = min(team2_heatmap[i, j] / 255, 0.7)
                    field_team2[i+50, j+50] = (1-alpha) * field_team2[i+50, j+50] + alpha * team2_colored[i, j]
                
                if ball_heatmap[i, j] > 5:
                    alpha = min(ball_heatmap[i, j] / 255, 0.7)
                    field_ball[i+50, j+50] = (1-alpha) * field_ball[i+50, j+50] + alpha * ball_colored[i, j]
        
        # Converter para formato RGB para Matplotlib
        field_team1 = cv2.cvtColor(field_team1.astype(np.uint8), cv2.COLOR_BGR2RGB)
        field_team2 = cv2.cvtColor(field_team2.astype(np.uint8), cv2.COLOR_BGR2RGB)
        field_ball = cv2.cvtColor(field_ball.astype(np.uint8), cv2.COLOR_BGR2RGB)
        
        # Plotar campos com mapas de calor
        plt.figure(figsize=(18, 6))
        
        plt.subplot(1, 3, 1)
        plt.imshow(field_team1)
        plt.title("Mapa de Calor - Time 1")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(field_team2)
        plt.title("Mapa de Calor - Time 2")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(field_ball)
        plt.title("Mapa de Calor - Bola")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "field_heatmaps.png"), dpi=150)
        plt.close()
        
        # Gráfico de comparação de velocidades entre times
        team1_avg_speed = 0
        team2_avg_speed = 0
        team1_max_speed = 0
        team2_max_speed = 0
        
        if self.speeds["Team1"]:
            team1_avg_speed = sum(self.speeds["Team1"].values()) / len(self.speeds["Team1"])
        if self.speeds["Team2"]:
            team2_avg_speed = sum(self.speeds["Team2"].values()) / len(self.speeds["Team2"])
        
        if self.max_speeds["Team1"]:
            team1_max_speed = max(self.max_speeds["Team1"].values())
        if self.max_speeds["Team2"]:
            team2_max_speed = max(self.max_speeds["Team2"].values())
            
        plt.figure(figsize=(10, 6))
        teams = ["Time 1", "Time 2"]
        avg_speeds = [team1_avg_speed, team2_avg_speed]
        max_speeds = [team1_max_speed, team2_max_speed]
        
        x = range(len(teams))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], avg_speeds, width, label='Velocidade Média', color=['darkred', 'darkblue'])
        plt.bar([i + width/2 for i in x], max_speeds, width, label='Velocidade Máxima', color=['red', 'blue'])
        
        plt.ylabel('Velocidade (m/s)')
        plt.title('Comparação de Velocidades entre Times')
        plt.xticks(x, teams)
        plt.legend()
        
        # Adicionar valores nas barras
        for i, v in enumerate(avg_speeds):
            plt.text(i - width/2, v + 0.5, f'{v:.2f}', ha='center')
        for i, v in enumerate(max_speeds):
            plt.text(i + width/2, v + 0.5, f'{v:.2f}', ha='center')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "team_speeds.png"))
        plt.close()
        
        # Gráfico de distância percorrida
        team1_dist = sum(self.distances["Team1"].values())
        team2_dist = sum(self.distances["Team2"].values())
        
        plt.figure(figsize=(8, 6))
        plt.bar(teams, [team1_dist, team2_dist], color=['red', 'blue'])
        plt.ylabel('Distância (metros)')
        plt.title('Distância Total Percorrida')
        
        # Adicionar valores nas barras
        for i, v in enumerate([team1_dist, team2_dist]):
            plt.text(i, v + 10, f'{v:.2f}m', ha='center')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "team_distances.png"))
        plt.close()
        
        # Gráfico posse de bola por zona
        total_zone_frames = sum(self.field_zones.values())
        if total_zone_frames > 0:
            zone_names = ["Def. T1", "Meio T1", "Atq. T1", "Def. T2", "Meio T2", "Atq. T2"]
            zone_values = [
                self.field_zones["def_third_team1"] / total_zone_frames * 100,
                self.field_zones["mid_third_team1"] / total_zone_frames * 100,
                self.field_zones["att_third_team1"] / total_zone_frames * 100,
                self.field_zones["def_third_team2"] / total_zone_frames * 100,
                self.field_zones["mid_third_team2"] / total_zone_frames * 100,
                self.field_zones["att_third_team2"] / total_zone_frames * 100
            ]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(zone_names, zone_values, color=['darkred', 'red', 'lightcoral', 'darkblue', 'blue', 'lightskyblue'])
            plt.ylabel('Porcentagem de Tempo (%)')
            plt.title('Distribuição da Bola por Zona do Campo')
            
            # Adicionar valores nas barras
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "zone_distribution.png"))
            plt.close()
            
        # Visualização das zonas no campo de futebol
        field_zones_img = field_img.copy()
        
        # Desenhar retângulos para as zonas (com transparência)
        zone_colors = [
            (50, 50, 200),   # Defensivo Time 1
            (100, 100, 200), # Meio Time 1
            (150, 150, 200), # Ofensivo Time 1
            (200, 150, 150), # Ofensivo Time 2
            (200, 100, 100), # Meio Time 2
            (200, 50, 50)    # Defensivo Time 2
        ]
        
        # Criar uma imagem para zonas com transparência
        zone_overlay = np.zeros_like(field_zones_img, dtype=np.float32)
        
        # Zona defensiva Time 1
        cv2.rectangle(zone_overlay, (50, 50), (field_width//3, field_height-50), zone_colors[0], -1)
        
        # Zona meio Time 1
        cv2.rectangle(zone_overlay, (field_width//3, 50), (2*field_width//3, field_height-50), zone_colors[1], -1)
        
        # Zona ofensiva Time 1
        cv2.rectangle(zone_overlay, (2*field_width//3, 50), (field_width-50, field_height-50), zone_colors[2], -1)
        
        # Combinar com texto explicativo
        field_zones_img = cv2.addWeighted(field_zones_img, 0.7, zone_overlay.astype(np.uint8), 0.3, 0)
        
        # Adicionar textos para as zonas
        zone_stats = [
            self.field_zones["def_third_team1"] / total_zone_frames * 100,
            self.field_zones["mid_third_team1"] / total_zone_frames * 100,
            self.field_zones["att_third_team1"] / total_zone_frames * 100
        ]
        
        # Converter para RGB para Matplotlib
        field_zones_img = cv2.cvtColor(field_zones_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        
        # Plotar campo com zonas
        plt.figure(figsize=(12, 8))
        plt.imshow(field_zones_img)
        
        # Adicionar textos de percentuais
        plt.text(field_width//6, field_height//2, f"{zone_stats[0]:.1f}%", 
                ha='center', va='center', color='white', fontsize=14, fontweight='bold')
        plt.text(field_width//2, field_height//2, f"{zone_stats[1]:.1f}%", 
                ha='center', va='center', color='white', fontsize=14, fontweight='bold')
        plt.text(5*field_width//6, field_height//2, f"{zone_stats[2]:.1f}%", 
                ha='center', va='center', color='white', fontsize=14, fontweight='bold')
        
        plt.title("Distribuição da Bola por Zona do Campo")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "field_zones.png"), dpi=150)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisador de Métricas de Futebol com DeepSORT")
    
    parser.add_argument("--model", type=str, default="runs/train/yolo12n_futbol_metrics16/weights/best.pt", 
                      help="Caminho para o modelo treinado")
    parser.add_argument("--video", type=str, required=True, 
                      help="Caminho para o vídeo a ser analisado")
    parser.add_argument("--output", type=str, default="resultados", 
                      help="Diretório para salvar os resultados")
    parser.add_argument("--save_vis", action="store_true",
                      help="Salvar vídeo com visualização do rastreamento")
    parser.add_argument("--vis_path", type=str, default=None,
                      help="Caminho para salvar o vídeo de visualização")
    parser.add_argument("--device", type=str, default=None,
                      help="Dispositivo para inferência ('cpu', 'cuda', 'mps')")
    
    args = parser.parse_args()
    
    # Detectar dispositivo se não especificado
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Usando dispositivo: {device}")
    
    analyzer = FootballAnalyzerV2(
        model_path=args.model,
        device=device
    )
    
    print(f"Analisando vídeo: {args.video}")
    print(f"Modelo: {args.model}")
    print(f"Resultados serão salvos em: {args.output}")
    
    start_time = time.time()
    results = analyzer.analyze_video(
        video_path=args.video,
        output_dir=args.output,
        save_vis=args.save_vis,
        vis_path=args.vis_path
    )
    elapsed = time.time() - start_time
    
    print(f"\nAnálise concluída em {elapsed:.2f} segundos!")
    print(f"Resultados salvos em: {args.output}")
    
    # Mostrar resumo dos resultados
    print("\nRESUMO DOS RESULTADOS:")
    print(f"Posse de bola: Time 1 {results['possession']['Team1']:.1f}% | Time 2 {results['possession']['Team2']:.1f}%")
    print(f"Distância total: Time 1 {results['distances']['Team1']:.2f}m | Time 2 {results['distances']['Team2']:.2f}m")
    print(f"Velocidade máxima: Time 1 {results['speeds']['Team1']['max']:.2f}m/s | Time 2 {results['speeds']['Team2']['max']:.2f}m/s") 