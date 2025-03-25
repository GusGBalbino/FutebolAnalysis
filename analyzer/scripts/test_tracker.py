import cv2
import numpy as np
import argparse
import time
import sys
import os

# Adicionar o diretório raiz e o diretório de scripts ao sys.path para encontrar módulos
current_dir = os.path.dirname(os.path.abspath(__file__))
analyzer_dir = os.path.dirname(current_dir)
sys.path.append(analyzer_dir)
from scripts.football_tracker import FootballTracker

def main():
    parser = argparse.ArgumentParser(description="Testar rastreador de futebol")
    parser.add_argument("--model", type=str, default=os.path.join(analyzer_dir, "models/best.pt"),
                       help="Caminho para o modelo YOLO treinado")
    parser.add_argument("--video", type=str, required=True,
                       help="Caminho para o vídeo a ser analisado")
    parser.add_argument("--output", type=str, default=None,
                       help="Caminho para salvar o vídeo com visualização (opcional)")
    parser.add_argument("--conf", type=float, default=0.4,
                       help="Limiar de confiança para detecções")
    
    args = parser.parse_args()
    
    # Carregar o tracker
    print(f"Carregando modelo: {args.model}")
    tracker = FootballTracker(args.model)
    
    # Abrir vídeo
    print(f"Abrindo vídeo: {args.video}")
    cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {args.video}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Configurar saída de vídeo
    if args.output:
        print(f"Visualização será salva em: {args.output}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Processar vídeo
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        
        # Rastrear objetos no frame
        ball_pos = tracker.track_frame(frame, frame_idx, confidence_threshold=args.conf)
        
        # Desenhar visualizações
        vis_frame = tracker.draw_tracks(frame.copy(), ball_pos)
        
        process_time = time.time() - start_time
        fps_text = f"FPS: {1/process_time:.1f}"
        
        # Adicionar FPS e informações
        cv2.putText(vis_frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Mostrar frame
        cv2.imshow('Football Tracking', vis_frame)
        
        # Salvar frame
        if args.output:
            out.write(vis_frame)
            
        frame_idx += 1
        
        # Sair com ESC
        if cv2.waitKey(1) == 27:
            break
    
    # Limpar
    cap.release()
    if args.output:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"Processados {frame_idx} frames.")


if __name__ == "__main__":
    main() 