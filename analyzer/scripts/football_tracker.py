import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import math
import torch

class FieldCalibrator:
    """Classe para calibração do campo e conversão de pixels para metros"""
    
    def __init__(self, field_length=105, field_width=68):
        self.field_length = field_length  # Comprimento padrão de um campo de futebol em metros
        self.field_width = field_width    # Largura padrão de um campo de futebol em metros
        self.px_per_meter = None
        self.homography_matrix = None
        self.calibrated = False
        
    def calibrate_from_corners(self, corners_px):
        """
        Calibra usando as 4 coordenadas dos cantos do campo
        corners_px: lista de 4 pontos [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] em pixels
        """
        if len(corners_px) != 4:
            raise ValueError("São necessários exatamente 4 cantos para calibração")
            
        # Corners in pixel space (source)
        src_pts = np.array(corners_px, dtype=np.float32)
        
        # Corners in meter space (destination)
        # Ordem: Canto superior esquerdo, superior direito, inferior direito, inferior esquerdo
        dst_pts = np.array([
            [0, 0],
            [self.field_length, 0],
            [self.field_length, self.field_width],
            [0, self.field_width]
        ], dtype=np.float32)
        
        # Calcular matriz de homografia
        self.homography_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.calibrated = True
        
        # Calcular média de pixels por metro (aproximado)
        diag_px = math.sqrt((corners_px[0][0] - corners_px[2][0])**2 + (corners_px[0][1] - corners_px[2][1])**2)
        diag_m = math.sqrt(self.field_length**2 + self.field_width**2)
        self.px_per_meter = diag_px / diag_m
        
        return self.homography_matrix
    
    def calibrate_from_field_size(self, img_width, img_height):
        """Calibração simplificada baseada no tamanho da imagem (menos preciso)"""
        # Estimativa aproximada de pixels por metro
        self.px_per_meter = min(img_width / self.field_length, img_height / self.field_width)
        self.calibrated = True
        return self.px_per_meter
    
    def pixel_to_meter(self, point_px):
        """Converte coordenadas de pixel para metros"""
        if not self.calibrated:
            raise ValueError("O campo não foi calibrado. Chame 'calibrate_*' primeiro")
            
        if self.homography_matrix is not None:
            # Usando homografia (mais preciso)
            point = np.array([point_px[0], point_px[1], 1], dtype=np.float32).reshape(1, 1, 3)
            transformed = cv2.perspectiveTransform(point, self.homography_matrix)
            return (transformed[0][0][0], transformed[0][0][1])
        else:
            # Usando estimativa simples (menos preciso)
            return (point_px[0] / self.px_per_meter, point_px[1] / self.px_per_meter)
            
    def calc_distance_meters(self, point1_px, point2_px):
        """Calcula distância em metros entre dois pontos em pixels"""
        if not self.calibrated:
            raise ValueError("O campo não foi calibrado. Chame 'calibrate_*' primeiro")
            
        if self.homography_matrix is not None:
            # Conversão usando homografia
            p1_m = self.pixel_to_meter(point1_px)
            p2_m = self.pixel_to_meter(point2_px)
            return math.sqrt((p2_m[0] - p1_m[0])**2 + (p2_m[1] - p1_m[1])**2)
        else:
            # Estimativa baseada em pixels por metro
            dist_px = math.sqrt((point2_px[0] - point1_px[0])**2 + (point2_px[1] - point1_px[1])**2)
            return dist_px / self.px_per_meter


class BallTracker:
    """Classe especializada para rastreamento de bola"""
    
    def __init__(self, confidence_threshold=0.5, iou_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.ball_tracks = []
        self.kalman_filter = cv2.KalmanFilter(4, 2)  # Estado: x, y, dx, dy; Medida: x, y
        self.kalman_initialized = False
        self.last_detected_pos = None
        self.lost_frames = 0
        self.max_lost_frames = 10  # Máximo de frames para prever sem detecção
        
        # Configuração do filtro de Kalman
        self.kalman_filter.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        
        self.kalman_filter.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        
        # Ruído de processo (controla a suavidade)
        self.kalman_filter.processNoiseCov = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32) * 0.03
        
        # Ruído de medição (menor = mais confiança nas medições)
        self.kalman_filter.measurementNoiseCov = np.array([
            [1, 0],
            [0, 1]
        ], np.float32) * 0.1
    
    def initialize_kalman(self, x, y):
        """Inicializa o filtro de Kalman com a posição inicial"""
        self.kalman_filter.statePre = np.array([x, y, 0, 0], np.float32)
        self.kalman_filter.statePost = np.array([x, y, 0, 0], np.float32)
        self.kalman_initialized = True
        
    def update(self, ball_detections, frame_idx, frame=None):
        """Atualiza o rastreamento da bola usando detecções e filtro de Kalman"""
        if len(ball_detections) > 0:
            # Ordenar detecções por confiança
            best_detection = max(ball_detections, key=lambda x: x[4])
            x, y, w, h, conf = best_detection
            
            center_x, center_y = x + w/2, y + h/2
            self.last_detected_pos = (center_x, center_y)
            self.lost_frames = 0
            
            # Inicializar ou atualizar filtro Kalman
            if not self.kalman_initialized:
                self.initialize_kalman(center_x, center_y)
            else:
                measurement = np.array([[center_x], [center_y]], np.float32)
                self.kalman_filter.correct(measurement)
            
            # Predição
            prediction = self.kalman_filter.predict()
            predicted_x, predicted_y = prediction[0], prediction[1]
            
            # Adicionar ao histórico de rastreamento
            self.ball_tracks.append((frame_idx, (center_x, center_y), True))
            
            return (center_x, center_y), True
            
        elif self.kalman_initialized and self.lost_frames < self.max_lost_frames:
            # Bola não detectada, mas podemos prever com Kalman
            self.lost_frames += 1
            prediction = self.kalman_filter.predict()
            predicted_x, predicted_y = prediction[0], prediction[1]
            
            # Adicionar previsão ao histórico
            self.ball_tracks.append((frame_idx, (predicted_x, predicted_y), False))
            
            return (predicted_x, predicted_y), False
            
        return None, False
    
    def get_trajectory(self, n_frames=10):
        """Obtém a trajetória recente da bola para visualização"""
        if len(self.ball_tracks) < 2:
            return []
            
        return [pos for _, pos, _ in self.ball_tracks[-n_frames:]]


class FootballTracker:
    """Classe principal para rastrear jogadores e bola no futebol"""
    
    def __init__(self, model_path, device='cpu'):
        # Modelos e rastreadores
        self.model = YOLO(model_path)
        self.device = device
        
        # DeepSORT para jogadores
        self.tracker = DeepSort(
            max_age=30,                     # Máximo de frames para manter ID sem detecção
            n_init=3,                       # Frames antes de considerar uma track válida
            max_iou_distance=0.7,           # Limiar IOU para associação
            max_cosine_distance=0.2,        # Limiar de aparência para associação
            nn_budget=100                   # Número máximo de exemplos por identidade
        )
        
        # Rastreador especializado para bola
        self.ball_tracker = BallTracker(confidence_threshold=0.4, iou_threshold=0.3)
        
        # Dicionários para armazenar trajetórias
        self.team1_tracks = {}  # {id: [(frame_idx, pos_x, pos_y), ...]}
        self.team2_tracks = {}  
        self.gk1_tracks = {}
        self.gk2_tracks = {}
        self.referee_tracks = {}
        
        # Mapa de classes
        self.class_map = {
            0: "Ball",
            1: "GK1",
            2: "GK2", 
            3: "Referee",
            4: "Team1",
            5: "Team2"
        }
    
    def reset_tracks(self):
        """Limpa todos os dados de rastreamento armazenados"""
        self.team1_tracks = {}
        self.team2_tracks = {}
        self.gk1_tracks = {}
        self.gk2_tracks = {}
        self.referee_tracks = {}
        self.ball_tracker.ball_tracks = []
        self.ball_tracker.kalman_initialized = False
        self.ball_tracker.last_detected_pos = None
        self.ball_tracker.lost_frames = 0
        
    def track_frame(self, frame, frame_idx, confidence_threshold=0.5):
        """
        Rastreia jogadores e bola em um frame
        
        Args:
            frame: Imagem do frame
            frame_idx: Índice do frame
            confidence_threshold: Limite mínimo de confiança
            
        Returns:
            tracked_objects: Lista de objetos rastreados com formato
                            [(classe, id, bbox, conf), ...]
            ball_pos: Posição da bola (x, y) ou None se não detectada
        """
        # Rodar detecção YOLO
        results = self.model(frame, verbose=False)[0]
        
        detections = []           # Para DeepSORT [(x1,y1,x2,y2), class_id, conf]
        ball_detections = []      # Para o rastreador de bola [x,y,w,h,conf]
        
        # Processa detecções
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() 
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                # Filtro de confiança
                if conf < confidence_threshold:
                    continue
                
                # Separar bola para rastreador especializado
                if cls_id == 0:  # Bola
                    w, h = x2 - x1, y2 - y1
                    ball_detections.append([x1, y1, w, h, conf])
                else:
                    # Outros objetos vão para o DeepSORT
                    detections.append(([x1, y1, x2, y2], cls_id, conf))
        
        # Rastrear bola com rastreador especializado
        ball_pos, ball_detected = self.ball_tracker.update(ball_detections, frame_idx, frame)
        
        # Rastrear jogadores com DeepSORT
        if detections:
            # Formato correto para DeepSORT: lista de tuplas (bbox, conf, cls_id)
            # Onde bbox deve estar no formato [x1, y1, w, h]
            detections_for_deepsort = []
            
            for det in detections:
                bbox, cls_id, conf = det
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                detections_for_deepsort.append(([x1, y1, w, h], conf, cls_id))
            
            # Atualizar tracks
            tracks = self.tracker.update_tracks(detections_for_deepsort, frame=frame)
            
            # Atualizar trajetórias
            for track in tracks:
                if not track.is_confirmed():
                    continue
                    
                track_id = track.track_id
                ltrb = track.to_ltrb()
                class_id = track.det_class
                
                bbox = (ltrb[0], ltrb[1], ltrb[2], ltrb[3])  # x1,y1,x2,y2
                center_x = (ltrb[0] + ltrb[2]) / 2
                center_y = (ltrb[1] + ltrb[3]) / 2
                
                # Armazenar trajetória baseada na classe
                if class_id == 1:  # GK1
                    if track_id not in self.gk1_tracks:
                        self.gk1_tracks[track_id] = []
                    self.gk1_tracks[track_id].append((frame_idx, (center_x, center_y)))
                elif class_id == 2:  # GK2
                    if track_id not in self.gk2_tracks:
                        self.gk2_tracks[track_id] = []
                    self.gk2_tracks[track_id].append((frame_idx, (center_x, center_y)))
                elif class_id == 3:  # Referee
                    if track_id not in self.referee_tracks:
                        self.referee_tracks[track_id] = []
                    self.referee_tracks[track_id].append((frame_idx, (center_x, center_y)))
                elif class_id == 4:  # Team1
                    if track_id not in self.team1_tracks:
                        self.team1_tracks[track_id] = []
                    self.team1_tracks[track_id].append((frame_idx, (center_x, center_y)))
                elif class_id == 5:  # Team2
                    if track_id not in self.team2_tracks:
                        self.team2_tracks[track_id] = []
                    self.team2_tracks[track_id].append((frame_idx, (center_x, center_y)))
        
        return ball_pos
    
    def draw_tracks(self, frame, ball_pos=None, track_history=10):
        """
        Desenha as trajetórias e caixas de detecção no frame
        
        Args:
            frame: Imagem do frame
            ball_pos: Posição da bola
            track_history: Número de frames de histórico para desenhar
            
        Returns:
            frame: Frame com visualizações
        """
        # Desenhar jogadores do Time 1
        for track_id, track in self.team1_tracks.items():
            if len(track) > 0:
                # Últimas posições conhecidas
                positions = [pos for _, pos in track[-track_history:]]
                
                # Desenhar trajetória
                if len(positions) > 1:
                    for i in range(1, len(positions)):
                        cv2.line(frame, 
                                (int(positions[i-1][0]), int(positions[i-1][1])),
                                (int(positions[i][0]), int(positions[i][1])),
                                (0, 0, 255), 2)
                
                # Desenhar último ponto
                if positions:
                    cv2.circle(frame, (int(positions[-1][0]), int(positions[-1][1])), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"T1-{track_id}", 
                                (int(positions[-1][0]), int(positions[-1][1])-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Desenhar jogadores do Time 2 (azul)
        for track_id, track in self.team2_tracks.items():
            if len(track) > 0:
                positions = [pos for _, pos in track[-track_history:]]
                if len(positions) > 1:
                    for i in range(1, len(positions)):
                        cv2.line(frame, 
                                (int(positions[i-1][0]), int(positions[i-1][1])),
                                (int(positions[i][0]), int(positions[i][1])),
                                (255, 0, 0), 2)
                
                if positions:
                    cv2.circle(frame, (int(positions[-1][0]), int(positions[-1][1])), 5, (255, 0, 0), -1)
                    cv2.putText(frame, f"T2-{track_id}", 
                                (int(positions[-1][0]), int(positions[-1][1])-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Desenhar goleiros (verde)
        gk_tracks = {**self.gk1_tracks, **self.gk2_tracks}
        for track_id, track in gk_tracks.items():
            if len(track) > 0:
                positions = [pos for _, pos in track[-track_history:]]
                if len(positions) > 1:
                    for i in range(1, len(positions)):
                        cv2.line(frame, 
                                (int(positions[i-1][0]), int(positions[i-1][1])),
                                (int(positions[i][0]), int(positions[i][1])),
                                (0, 255, 0), 2)
                
                if positions:
                    cv2.circle(frame, (int(positions[-1][0]), int(positions[-1][1])), 5, (0, 255, 0), -1)
                    cv2.putText(frame, f"GK-{track_id}", 
                                (int(positions[-1][0]), int(positions[-1][1])-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Desenhar árbitro (amarelo)
        for track_id, track in self.referee_tracks.items():
            if len(track) > 0:
                positions = [pos for _, pos in track[-track_history:]]
                if len(positions) > 1:
                    for i in range(1, len(positions)):
                        cv2.line(frame, 
                                (int(positions[i-1][0]), int(positions[i-1][1])),
                                (int(positions[i][0]), int(positions[i][1])),
                                (0, 255, 255), 2)
                
                if positions:
                    cv2.circle(frame, (int(positions[-1][0]), int(positions[-1][1])), 5, (0, 255, 255), -1)
                    cv2.putText(frame, f"Ref-{track_id}", 
                                (int(positions[-1][0]), int(positions[-1][1])-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Desenhar a bola (branco) com sua trajetória
        if ball_pos is not None:
            # Obter trajetória recente
            ball_trajectory = self.ball_tracker.get_trajectory(track_history)
            
            # Desenhar trajetória
            if len(ball_trajectory) > 1:
                for i in range(1, len(ball_trajectory)):
                    cv2.line(frame, 
                            (int(ball_trajectory[i-1][0]), int(ball_trajectory[i-1][1])),
                            (int(ball_trajectory[i][0]), int(ball_trajectory[i][1])),
                            (255, 255, 255), 2)
            
            # Desenhar posição atual
            cv2.circle(frame, (int(ball_pos[0]), int(ball_pos[1])), 5, (255, 255, 255), -1)
            cv2.putText(frame, "Ball", 
                        (int(ball_pos[0]), int(ball_pos[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame 