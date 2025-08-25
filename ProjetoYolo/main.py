import cv2
import numpy as np
import argparse
import time
import os
import sys
from typing import Optional, Dict, Any

from shoe_detector import ShoeDetector
from roi_manager import ROIManager
from event_tracker import EventTracker
from utils import (
    setup_logging, create_output_directories, save_results,
    create_summary_plot, format_time, calculate_fps
)
from config import (
    VIDEO_CONFIG, UI_CONFIG, OUTPUT_CONFIG, COLORS,
    PERFORMANCE_CONFIG
)


class ShoeDetectionSystem:

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.video_config = VIDEO_CONFIG
        self.ui_config = UI_CONFIG
        self.output_config = OUTPUT_CONFIG
        self.performance_config = PERFORMANCE_CONFIG

        self.shoe_detector = None
        self.roi_manager = None
        self.event_tracker = None

        self.is_running = False
        self.is_paused = False
        self.current_frame = 0
        self.start_time = None
        self.frame_count = 0

        self.frame_history = []
        self.performance_metrics = {}

        self.logger = setup_logging(
            log_level='INFO',
            log_file='outputs/detection.log',
            console_output=True
        )

        create_output_directories()

        self._initialize_components()

    def _initialize_components(self) -> None:
        try:
            self.logger.info("Inicializando componentes do sistema...")

            self.shoe_detector = ShoeDetector()
            self.logger.info("Detector de tênis inicializado")

            self.roi_manager = ROIManager()
            self.logger.info("Gerenciador de ROI inicializado")

            self.event_tracker = EventTracker()
            self.logger.info("Rastreador de eventos inicializado")

            self.logger.info("Sistema inicializado com sucesso")

        except Exception as e:
            self.logger.error(f"Erro ao inicializar componentes: {e}")
            raise

    def process_video(self, video_path: str) -> Dict[str, Any]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

        self.logger.info(f"Iniciando processamento do vídeo: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.logger.info(f"Vídeo: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")

        output_path = f"outputs/processed_{os.path.basename(video_path)}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        self.start_time = time.time()
        self.frame_count = 0
        self.frame_history = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = self._process_frame(frame, self.frame_count)

                out.write(processed_frame)

                self.frame_count += 1

                if self.frame_count % 30 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    elapsed = time.time() - self.start_time
                    fps_current = self.frame_count / elapsed
                    self.logger.info(
                        f"Progresso: {progress:.1f}% - Frame {self.frame_count}/{total_frames} - FPS: {fps_current:.1f}")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()

        results = self._generate_final_results()

        self._save_results(results, output_path)

        self.logger.info("Processamento do vídeo concluído")
        return results

    def process_camera(self, camera_id: int = 0) -> None:
        self.logger.info(f"Iniciando processamento da câmera {camera_id}")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir a câmera {camera_id}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_config['frame_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_config['frame_height'])
        cap.set(cv2.CAP_PROP_FPS, self.video_config['fps_target'])

        self.start_time = time.time()
        self.frame_count = 0
        self.is_running = True

        cv2.namedWindow('Sistema de Detecção de Tênis')
        cv2.setMouseCallback('Sistema de Detecção de Tênis',
                             self.roi_manager.handle_mouse_event)

        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Erro ao ler frame da câmera")
                    continue

                processed_frame = self._process_frame(frame, self.frame_count)

                cv2.imshow('Sistema de Detecção de Tênis', processed_frame)

                self.frame_count += 1

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    self.is_paused = not self.is_paused
                    self.logger.info("Sistema pausado" if self.is_paused else "Sistema retomado")
                elif key == ord('r'):
                    self._reset_system()
                elif key == ord('c'):
                    self.roi_manager.start_manual_config()
                elif key == ord('s'):
                    self._save_current_state()

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.is_running = False

        results = self._generate_final_results()
        self._save_results(results, "outputs/camera_results.json")

        self.logger.info("Processamento da câmera concluído")

    def _process_frame(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        if self.is_paused:
            return frame

        target_width = self.video_config['frame_width']
        target_height = self.video_config['frame_height']

        if frame.shape[1] != target_width or frame.shape[0] != target_height:
            frame = cv2.resize(frame, (target_width, target_height))

        detections = self.shoe_detector.detect_shoes(frame)

        roi_detections = []
        for detection in detections:
            if self.roi_manager.is_bbox_in_roi(detection['bbox']):
                roi_detections.append(detection)

        tracking_info = self.event_tracker.update_tracking(roi_detections, frame_number)

        for obj_id, obj_info in tracking_info['current_objects'].items():
            for detection in roi_detections:
                if (detection['bbox'] == obj_info['bbox']).all():
                    detection['id'] = obj_id
                    break

        result_frame = frame.copy()

        if self.ui_config['show_roi']:
            result_frame = self.roi_manager.draw_roi(result_frame)

        if roi_detections:
            result_frame = self.shoe_detector.draw_detections(
                result_frame, roi_detections,
                show_confidence=self.ui_config['show_confidence'],
                show_tracking=self.ui_config['show_tracking']
            )

        result_frame = self._draw_ui_overlay(result_frame, tracking_info, frame_number)

        self.frame_history.append({
            'frame': frame_number,
            'count': len(roi_detections),
            'detections': roi_detections,
            'tracking_info': tracking_info
        })

        return result_frame

    def _draw_ui_overlay(self, frame: np.ndarray, tracking_info: Dict[str, Any],
                         frame_number: int) -> np.ndarray:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.ui_config['font_scale']
        font_thickness = self.ui_config['font_thickness']
        text_color = COLORS['text']
        bg_color = COLORS['background']

        if self.start_time:
            fps = calculate_fps(self.start_time, self.frame_count)
        else:
            fps = 0

        info_lines = [
            f"Frame: {frame_number}",
            f"FPS: {fps:.1f}",
            f"Tênis Detectados: {tracking_info['total_count']}",
            f"Total Coletados: {tracking_info['collected_count']}",
            f"Novos: {tracking_info['new_count']}"
        ]

        y_offset = 30
        for i, line in enumerate(info_lines):
            text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
            text_width = text_size[0]
            text_height = text_size[1]

            cv2.rectangle(frame, (10, y_offset - text_height - 5),
                          (10 + text_width + 10, y_offset + 5), bg_color, -1)

            cv2.putText(frame, line, (15, y_offset), font, font_scale,
                        text_color, font_thickness)

            y_offset += text_height + 10

        instructions = [
            "P: Pausar/Retomar | R: Resetar | C: Configurar ROI | Q: Sair"
        ]

        y_offset += 20
        for instruction in instructions:
            text_size = cv2.getTextSize(instruction, font, 0.5, 1)[0]
            cv2.rectangle(frame, (10, y_offset - text_size[1] - 5),
                          (10 + text_size[0] + 10, y_offset + 5), bg_color, -1)
            cv2.putText(frame, instruction, (15, y_offset), font, 0.5, text_color, 1)
            y_offset += text_size[1] + 10

        return frame

    def _generate_final_results(self) -> Dict[str, Any]:
        tracking_summary = self.event_tracker.get_tracking_summary()

        detection_stats = self.shoe_detector.get_detection_statistics()

        total_time = time.time() - self.start_time if self.start_time else 0
        avg_fps = self.frame_count / total_time if total_time > 0 else 0

        self.performance_metrics = {
            'total_time': total_time,
            'total_frames': self.frame_count,
            'avg_fps': avg_fps,
            'processing_time_per_frame': total_time / self.frame_count if self.frame_count > 0 else 0
        }

        results = {
            'system_info': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'version': '1.0.0',
                'model_info': self.shoe_detector.get_model_info()
            },
            'tracking_summary': tracking_summary,
            'detection_statistics': detection_stats,
            'performance_metrics': self.performance_metrics,
            'frame_history': self.frame_history,
            'confidence_scores': self.shoe_detector.confidence_scores,
            'roi_info': {
                'roi': self.roi_manager.get_roi(),
                'roi_area': self.roi_manager.get_roi_area()
            }
        }

        return results

    def _save_results(self, results: Dict[str, Any], output_path: str) -> None:
        try:
            json_path = output_path.replace('.mp4', '.json')
            save_results(results, json_path)

            plot_path = output_path.replace('.mp4', '_summary.png')
            create_summary_plot(results, plot_path)

            model_config_path = 'outputs/model_config.json'
            self.shoe_detector.save_model_config(model_config_path)

            self.logger.info(f"Resultados salvos em: {json_path}")
            self.logger.info(f"Gráfico salvo em: {plot_path}")

        except Exception as e:
            self.logger.error(f"Erro ao salvar resultados: {e}")

    def _reset_system(self) -> None:
        self.logger.info("Resetando sistema...")

        self.event_tracker.reset_tracking()
        self.shoe_detector.reset_statistics()

        self.frame_count = 0
        self.start_time = time.time()
        self.frame_history.clear()

        self.logger.info("Sistema resetado")

    def _save_current_state(self) -> None:
        try:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            state_path = f"outputs/system_state_{timestamp}.json"

            current_state = {
                'timestamp': timestamp,
                'frame_count': self.frame_count,
                'tracking_summary': self.event_tracker.get_tracking_summary(),
                'detection_stats': self.shoe_detector.get_detection_statistics(),
                'roi_info': {
                    'roi': self.roi_manager.get_roi(),
                    'roi_area': self.roi_manager.get_roi_area()
                }
            }

            save_results(current_state, state_path)
            self.logger.info(f"Estado salvo em: {state_path}")

        except Exception as e:
            self.logger.error(f"Erro ao salvar estado: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Sistema de Detecção e Contagem de Tênis no Chão'
    )

    parser.add_argument(
        '--video',
        type=str,
        help='Caminho para arquivo de vídeo'
    )

    parser.add_argument(
        '--camera',
        type=int,
        default=None,
        help='ID da câmera (padrão: 0)'
    )

    parser.add_argument(
        '--roi-config',
        action='store_true',
        help='Configurar ROI manualmente'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Arquivo de configuração personalizado'
    )

    args = parser.parse_args()

    try:
        system = ShoeDetectionSystem()

        if args.roi_config:
            print("Modo de configuração de ROI")
            print("Pressione 'c' para iniciar configuração manual")
            print("Pressione 'r' para resetar ROI")
            print("Pressione 'q' para sair")

            cap = cv2.VideoCapture(0)
            cv2.namedWindow('Configuração de ROI')
            cv2.setMouseCallback('Configuração de ROI',
                                 system.roi_manager.handle_mouse_event)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = system.roi_manager.draw_roi(frame)

                if system.roi_manager.is_configuring:
                    frame = system.roi_manager.draw_preview_roi(frame)

                cv2.imshow('Configuração de ROI', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    system.roi_manager.reset_roi()
                elif key == ord('c'):
                    system.roi_manager.start_manual_config()

            cap.release()
            cv2.destroyAllWindows()

        elif args.video:
            results = system.process_video(args.video)

            print("\n=== RESUMO DOS RESULTADOS ===")
            print(f"Total de tênis detectados: {results['tracking_summary']['total_detected']}")
            print(f"Total coletados: {results['tracking_summary']['total_collected']}")
            print(f"Restantes: {results['tracking_summary']['current_count']}")
            print(f"Tempo de processamento: {format_time(results['performance_metrics']['total_time'])}")
            print(f"FPS médio: {results['performance_metrics']['avg_fps']:.2f}")

        elif args.camera is not None:
            system.process_camera(args.camera)

        else:
            print("Iniciando sistema com câmera padrão...")
            system.process_camera(0)

    except KeyboardInterrupt:
        print("\nSistema interrompido pelo usuário")
    except Exception as e:
        print(f"Erro no sistema: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
