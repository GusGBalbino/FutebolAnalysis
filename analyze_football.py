from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import math

class FootballAnalyzer:
    def __init__(self, model_path, video_fps=30.0, field_length=105, field_width=68):
        """
        Inicializa o analisador de futebol
        
        Args:
            model_path: Caminho para o modelo treinado
            video_fps: Frames por segundo do vídeo
            field_length: Comprimento do campo em metros
            field_width: Largura do campo em metros
        """
        self.model = YOLO(model_path)
        self.fps = video_fps
        self.field_length = field_length
        self.field_width = field_width
        
        # Dicionários para armazenar dados de rastreamento
        self.team1_tracks = defaultdict(list)
        self.team2_tracks = defaultdict(list)
        self.ball_tracks = []
        
        # Métricas
        self.possession = {"Team1": 0, "Team2": 0}
        self.distances = {"Team1": {}, "Team2": {}}
        self.speeds = {"Team1": {}, "Team2": {}}
        self.heatmaps = {"Team1": np.zeros((100, 100)), "Team2": np.zeros((100, 100))}
        
        # Mapa de classes
        self.class_map = {
            0: "Ball",
            1: "GK1",
            2: "GK2",
            3: "Refree",
            4: "Team1",
            5: "Team2"
        }
        
        # Parâmetros de pixel para conversão de metros
        self.px_to_meter = None
    
    def _update_possession(self, track_results, frame_idx):
        """Atualiza estatísticas de posse de bola"""
        ball_detected = False
        ball_pos = None
        
        # Encontrar posição da bola
        for r in track_results:
            if r.boxes.cls[0].item() == 0:  # Bola
                ball_detected = True
                ball_pos = (r.boxes.xywh[0][0].item(), r.boxes.xywh[0][1].item())
                self.ball_tracks.append((frame_idx, ball_pos))
                break
        
        if not ball_detected or ball_pos is None:
            return
            
        # Encontrar o jogador mais próximo da bola
        min_dist = float('inf')
        closest_team = None
        
        for r in track_results:
            cls = int(r.boxes.cls[0].item())
            if cls in [4, 5]:  # Team1 ou Team2
                player_pos = (r.boxes.xywh[0][0].item(), r.boxes.xywh[0][1].item())
                dist = math.sqrt((ball_pos[0] - player_pos[0])**2 + (ball_pos[1] - player_pos[1])**2)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_team = "Team1" if cls == 4 else "Team2"
        
        # Atualizar posse se um time estiver próximo o suficiente da bola
        if closest_team and min_dist < 50:  # Limiar de 50 pixels
            self.possession[closest_team] += 1
    
    def _update_player_tracking(self, track_results, frame_idx):
        """Atualiza o rastreamento de jogadores"""
        for r in track_results:
            cls = int(r.boxes.cls[0].item())
            track_id = int(r.boxes.id[0].item()) if r.boxes.id is not None else -1
            
            if track_id == -1:
                continue
                
            pos = (r.boxes.xywh[0][0].item(), r.boxes.xywh[0][1].item())
            
            if cls == 4:  # Team1
                self.team1_tracks[track_id].append((frame_idx, pos))
            elif cls == 5:  # Team2
                self.team2_tracks[track_id].append((frame_idx, pos))
    
    def _calculate_distances_speeds(self):
        """Calcula distâncias percorridas e velocidades dos jogadores"""
        # Processar Time 1
        for player_id, tracks in self.team1_tracks.items():
            if len(tracks) < 2:
                continue
                
            total_distance = 0
            speeds = []
            
            for i in range(1, len(tracks)):
                prev_frame, prev_pos = tracks[i-1]
                curr_frame, curr_pos = tracks[i]
                
                # Distância em pixels
                dist_px = math.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                
                # Converter para metros se possível
                dist_m = dist_px
                if self.px_to_meter:
                    dist_m = dist_px * self.px_to_meter
                
                # Calcular tempo entre frames
                time_diff = (curr_frame - prev_frame) / self.fps
                
                if time_diff > 0:
                    # Velocidade em metros por segundo
                    speed = dist_m / time_diff
                    speeds.append(speed)
                
                total_distance += dist_m
            
            self.distances["Team1"][player_id] = total_distance
            
            if speeds:
                self.speeds["Team1"][player_id] = {
                    "avg": sum(speeds) / len(speeds),
                    "max": max(speeds)
                }
        
        # Processar Time 2 (mesmo processo)
        for player_id, tracks in self.team2_tracks.items():
            if len(tracks) < 2:
                continue
                
            total_distance = 0
            speeds = []
            
            for i in range(1, len(tracks)):
                prev_frame, prev_pos = tracks[i-1]
                curr_frame, curr_pos = tracks[i]
                
                dist_px = math.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                
                dist_m = dist_px
                if self.px_to_meter:
                    dist_m = dist_px * self.px_to_meter
                
                time_diff = (curr_frame - prev_frame) / self.fps
                
                if time_diff > 0:
                    speed = dist_m / time_diff
                    speeds.append(speed)
                
                total_distance += dist_m
            
            self.distances["Team2"][player_id] = total_distance
            
            if speeds:
                self.speeds["Team2"][player_id] = {
                    "avg": sum(speeds) / len(speeds),
                    "max": max(speeds)
                }
    
    def _update_heatmaps(self, img_width, img_height):
        """Cria mapas de calor para ambos os times"""
        heatmap1 = np.zeros((100, 100))
        heatmap2 = np.zeros((100, 100))
        
        # Processar Time 1
        for player_id, tracks in self.team1_tracks.items():
            for _, pos in tracks:
                # Normalizar posição para grade 100x100
                x_norm = int((pos[0] / img_width) * 100)
                y_norm = int((pos[1] / img_height) * 100)
                
                # Limitar dentro da grade
                x_norm = max(0, min(99, x_norm))
                y_norm = max(0, min(99, y_norm))
                
                heatmap1[y_norm, x_norm] += 1
        
        # Processar Time 2
        for player_id, tracks in self.team2_tracks.items():
            for _, pos in tracks:
                x_norm = int((pos[0] / img_width) * 100)
                y_norm = int((pos[1] / img_height) * 100)
                
                x_norm = max(0, min(99, x_norm))
                y_norm = max(0, min(99, y_norm))
                
                heatmap2[y_norm, x_norm] += 1
        
        self.heatmaps["Team1"] = heatmap1
        self.heatmaps["Team2"] = heatmap2
    
    def analyze_video(self, video_path, output_dir="results"):
        """
        Analisa um vídeo de futebol e extrai métricas
        
        Args:
            video_path: Caminho para o vídeo
            output_dir: Diretório para salvar os resultados
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Estimativa grosseira de pixels para metros (baseada no tamanho padrão do campo)
        self.px_to_meter = self.field_length / img_width
        
        # Processar frames
        frame_idx = 0
        
        for frame_idx in tqdm(range(total_frames), desc="Analisando vídeo"):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Executar detecção e rastreamento
            results = self.model.track(frame, persist=True, verbose=False)
            
            if results[0].boxes.id is not None:
                # Atualizar rastreamento de jogadores
                self._update_player_tracking(results, frame_idx)
                
                # Atualizar estatísticas de posse de bola
                self._update_possession(results, frame_idx)
            
            frame_idx += 1
        
        cap.release()
        
        # Calcular métricas finais
        self._calculate_distances_speeds()
        self._update_heatmaps(img_width, img_height)
        
        # Gerar relatório
        self._generate_report(output_dir)
        
        return {
            "possession": self.possession,
            "distances": self.distances,
            "speeds": self.speeds,
            "heatmaps": self.heatmaps
        }
    
    def _generate_report(self, output_dir):
        """Gera um relatório com as métricas calculadas"""
        report_path = os.path.join(output_dir, "football_metrics_report.txt")
        
        with open(report_path, "w") as f:
            # Estatísticas de posse de bola
            total_possession = sum(self.possession.values())
            if total_possession > 0:
                team1_pct = (self.possession["Team1"] / total_possession) * 100
                team2_pct = (self.possession["Team2"] / total_possession) * 100
            else:
                team1_pct = team2_pct = 0
                
            f.write("=== RELATÓRIO DE ANÁLISE DE FUTEBOL ===\n\n")
            
            f.write("POSSE DE BOLA:\n")
            f.write(f"Time 1: {team1_pct:.2f}%\n")
            f.write(f"Time 2: {team2_pct:.2f}%\n\n")
            
            # Distância total percorrida
            team1_total_dist = sum(self.distances["Team1"].values())
            team2_total_dist = sum(self.distances["Team2"].values())
            
            f.write("DISTÂNCIA PERCORRIDA:\n")
            f.write(f"Time 1 (total): {team1_total_dist:.2f} metros\n")
            f.write(f"Time 2 (total): {team2_total_dist:.2f} metros\n\n")
            
            # Velocidades
            if self.speeds["Team1"]:
                team1_avg_speeds = [s["avg"] for s in self.speeds["Team1"].values()]
                team1_max_speeds = [s["max"] for s in self.speeds["Team1"].values()]
                
                f.write("VELOCIDADES (TIME 1):\n")
                f.write(f"Velocidade média: {sum(team1_avg_speeds)/len(team1_avg_speeds):.2f} m/s\n")
                f.write(f"Velocidade máxima: {max(team1_max_speeds):.2f} m/s\n\n")
            
            if self.speeds["Team2"]:
                team2_avg_speeds = [s["avg"] for s in self.speeds["Team2"].values()]
                team2_max_speeds = [s["max"] for s in self.speeds["Team2"].values()]
                
                f.write("VELOCIDADES (TIME 2):\n")
                f.write(f"Velocidade média: {sum(team2_avg_speeds)/len(team2_avg_speeds):.2f} m/s\n")
                f.write(f"Velocidade máxima: {max(team2_max_speeds):.2f} m/s\n\n")
            
            f.write("Relatório gerado em: " + time.strftime("%Y-%m-%d %H:%M:%S"))
        
        # Gerar gráficos
        self._generate_plots(output_dir)
        
        print(f"Relatório salvo em: {report_path}")
    
    def _generate_plots(self, output_dir):
        """Gera visualizações das métricas"""
        # Gráfico de posse de bola
        total_possession = sum(self.possession.values())
        if total_possession > 0:
            plt.figure(figsize=(10, 6))
            plt.pie(
                [self.possession["Team1"], self.possession["Team2"]],
                labels=["Time 1", "Time 2"],
                autopct='%1.1f%%',
                startangle=90,
                colors=['red', 'blue']
            )
            plt.title("Posse de Bola")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "possession.png"))
            plt.close()
        
        # Mapas de calor
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(self.heatmaps["Team1"], cmap="hot", interpolation="nearest")
        plt.title("Mapa de Calor - Time 1")
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(self.heatmaps["Team2"], cmap="hot", interpolation="nearest")
        plt.title("Mapa de Calor - Time 2")
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "heatmaps.png"))
        plt.close()


# Exemplo de uso
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analisador de Métricas de Futebol")
    parser.add_argument("--model", type=str, default="runs/train/yolo12n_futbol_metrics/weights/best.pt", 
                      help="Caminho para o modelo treinado")
    parser.add_argument("--video", type=str, required=True, 
                      help="Caminho para o vídeo a ser analisado")
    parser.add_argument("--output", type=str, default="results", 
                      help="Diretório para salvar os resultados")
    parser.add_argument("--fps", type=float, default=30.0, 
                      help="FPS do vídeo")
    
    args = parser.parse_args()
    
    analyzer = FootballAnalyzer(
        model_path=args.model,
        video_fps=args.fps
    )
    
    results = analyzer.analyze_video(
        video_path=args.video,
        output_dir=args.output
    )
    
    print("\nMétricas extraídas com sucesso!")
    print(f"Resultados salvos em: {args.output}") 