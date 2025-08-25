import cv2
import numpy as np
import os
import time
from typing import List, Dict, Any
from shoe_detector import ShoeDetector
from roi_manager import ROIManager
from event_tracker import EventTracker
from utils import draw_bbox, create_output_directories


class ShoeDetectionDemo:

    def __init__(self):
        self.shoe_detector = ShoeDetector()
        self.roi_manager = ROIManager()
        self.event_tracker = EventTracker()

        create_output_directories()

        print("Sistema de Demonstração inicializado!")
        print("Modelo YOLO carregado com sucesso.")

    def demo_image(self, image_path: str, save_result: bool = True) -> Dict[str, Any]:

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

        print(f"\nProcessando imagem: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Não foi possível carregar a imagem: {image_path}")

        start_time = time.time()
        detections = self.shoe_detector.detect_shoes(image)
        detection_time = time.time() - start_time

        print(f"Tempo de detecção: {detection_time:.3f}s")
        print(f"Tênis detectados: {len(detections)}")

        roi_detections = []
        for detection in detections:
            if self.roi_manager.is_bbox_in_roi(detection['bbox']):
                roi_detections.append(detection)

        print(f"Tênis na ROI: {len(roi_detections)}")

        result_image = image.copy()

        result_image = self.roi_manager.draw_roi(result_image)

        for detection in roi_detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']

            if confidence >= 0.8:
                color = (0, 255, 0)
            elif confidence >= 0.6:
                color = (255, 0, 0)
            else:
                color = (128, 128, 128)

            cv2.rectangle(result_image,
                          (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])),
                          color, 2)

            label = f"{class_name} {confidence:.2f}"
            cv2.putText(result_image, label,
                        (int(bbox[0]), int(bbox[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        info_text = [
            f"Tênis Detectados: {len(roi_detections)}",
            f"Tempo: {detection_time:.3f}s",
            f"Confiança Média: {np.mean([d['confidence'] for d in roi_detections]):.2f}" if roi_detections else "N/A"
        ]

        y_offset = 30
        for text in info_text:
            cv2.putText(result_image, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 25

        if save_result:
            output_path = f"outputs/demo_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, result_image)
            print(f"Resultado salvo em: {output_path}")

        cv2.imshow('Demonstração - Detecção de Tênis', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return {
            'total_detections': len(detections),
            'roi_detections': len(roi_detections),
            'detection_time': detection_time,
            'detections': roi_detections
        }

    def create_demo_video(self, output_path: str = "outputs/demo_video.mp4",
                          duration: int = 10, fps: int = 30) -> None:

        print(f"\nCriando vídeo de demonstração: {duration}s, {fps} FPS")

        width, height = 640, 480
        total_frames = duration * fps

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        shoes = [
            {'pos': [100, 300], 'size': 60, 'speed': [2, 0], 'visible': True},
            {'pos': [300, 350], 'size': 50, 'speed': [-1, 0], 'visible': True},
            {'pos': [500, 320], 'size': 55, 'speed': [0, -1], 'visible': True},
            {'pos': [200, 400], 'size': 45, 'speed': [1, 0], 'visible': True}
        ]

        roi = [50, 200, 590, 450]

        try:
            for frame_num in range(total_frames):

                frame = np.zeros((height, width, 3), dtype=np.uint8)

                cv2.rectangle(frame, (0, 200), (width, height), (50, 50, 50), -1)

                cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
                cv2.putText(frame, "ROI - Chão", (roi[0], roi[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                visible_shoes = 0
                for i, shoe in enumerate(shoes):
                    if not shoe['visible']:
                        continue

                    shoe['pos'][0] += shoe['speed'][0]
                    shoe['pos'][1] += shoe['speed'][1]

                    if (shoe['pos'][0] < 0 or shoe['pos'][0] > width or
                            shoe['pos'][1] < 200 or shoe['pos'][1] > height):

                        if frame_num > total_frames // 2:
                            shoe['visible'] = False
                        else:

                            shoe['pos'] = [np.random.randint(50, 590), np.random.randint(300, 400)]

                    x, y = int(shoe['pos'][0]), int(shoe['pos'][1])
                    size = shoe['size']

                    cv2.rectangle(frame, (x - size // 2, y - size // 2),
                                  (x + size // 2, y + size // 2), (0, 255, 0), -1)
                    cv2.rectangle(frame, (x - size // 2, y - size // 2),
                                  (x + size // 2, y + size // 2), (255, 255, 255), 2)

                    cv2.putText(frame, f"Tenis {i + 1}", (x - 20, y - size // 2 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    visible_shoes += 1

                info_text = [
                    f"Frame: {frame_num}/{total_frames}",
                    f"Tênis Visíveis: {visible_shoes}",
                    f"Tempo: {frame_num / fps:.1f}s"
                ]

                y_offset = 30
                for text in info_text:
                    cv2.putText(frame, text, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 20

                out.write(frame)

                if frame_num % 30 == 0:
                    progress = (frame_num / total_frames) * 100
                    print(f"Progresso: {progress:.1f}% - Frame {frame_num}/{total_frames}")

        finally:
            out.release()
        print(f"Vídeo de demonstração criado: {output_path}")
        print(f"Tênis visíveis no início: {len([s for s in shoes if s['visible']])}")
        print(f"Tênis coletados: {len([s for s in shoes if not s['visible']])}")

    def run_interactive_demo(self) -> None:
        print("\n=== DEMONSTRAÇÃO INTERATIVA ===")
        print("1. Testar com imagem")
        print("2. Criar vídeo de demonstração")
        print("3. Sair")

        while True:
            choice = input("\nEscolha uma opção (1-3): ").strip()

            if choice == '1':
                image_path = input("Digite o caminho da imagem: ").strip()
                if image_path:
                    try:
                        self.demo_image(image_path)
                    except Exception as e:
                        print(f"Erro: {e}")

            elif choice == '2':
                try:
                    duration = int(input("Duração do vídeo em segundos (padrão: 10): ") or "10")
                    fps = int(input("FPS do vídeo (padrão: 30): ") or "30")
                    self.create_demo_video(duration=duration, fps=fps)
                except ValueError:
                    print("Valor inválido. Usando valores padrão.")
                    self.create_demo_video()
                except Exception as e:
                    print(f"Erro: {e}")

            elif choice == '3':
                print("Demonstração finalizada.")
                break

            else:
                print("Opção inválida. Tente novamente.")

def main():
    try:
        demo = ShoeDetectionDemo()
        demo.run_interactive_demo()

    except KeyboardInterrupt:
        print("\nDemonstração interrompida pelo usuário")
    except Exception as e:
        print(f"Erro na demonstração: {e}")


if __name__ == "__main__":
    main()
