import cv2
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from ultralytics import YOLO
from config import MODEL_CONFIG, SHOE_DETECTION_CONFIG, COLORS, UI_CONFIG
from utils import (
    filter_detections_by_size, filter_detections_by_aspect_ratio,
    draw_bbox, calculate_fps, format_time
)


class ShoeDetector:

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or MODEL_CONFIG
        self.model = None
        self.model_size = self.config.get('model_size', 'n')
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.iou_threshold = self.config.get('iou_threshold', 0.45)
        self.max_det = self.config.get('max_det', 100)

        self.shoe_config = SHOE_DETECTION_CONFIG
        self.min_shoe_size = self.shoe_config.get('min_shoe_size', 20)
        self.max_shoe_size = self.shoe_config.get('max_shoe_size', 300)
        self.aspect_ratio_range = self.shoe_config.get('aspect_ratio_range', (0.5, 2.0))
        self.shoe_classes = self.shoe_config.get('class_names', ['shoe', 'sneaker', 'footwear'])

        self.total_detections = 0
        self.confidence_scores = []
        self.detection_times = []

        self._load_model()

    def _load_model(self) -> None:
        try:
            model_name = f"yolov8{self.model_size}.pt"
            print(f"Carregando modelo YOLO: {model_name}")

            self.model = YOLO(model_name)
            print(f"Modelo carregado com sucesso: {model_name}")

            device = "cuda" if self.model.device.type == "cuda" else "cpu"
            print(f"Dispositivo de inferência: {device}")

        except Exception as e:
            print(f"Erro ao carregar modelo YOLO: {e}")
            print("Tentando carregar modelo padrão...")
            try:
                self.model = YOLO('yolov8n.pt')
                print("Modelo padrão carregado com sucesso")
            except Exception as e2:
                print(f"Erro fatal ao carregar modelo: {e2}")
                raise

    def detect_shoes(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if self.model is None:
            print("Modelo não carregado")
            return []

        start_time = time.time()

        try:
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_det,
                verbose=False
            )

            detections = self._process_results(results[0], image.shape)

            filtered_detections = self._filter_shoe_detections(detections)

            detection_time = time.time() - start_time
            self.detection_times.append(detection_time)
            self.total_detections += len(filtered_detections)

            for det in filtered_detections:
                self.confidence_scores.append(det['confidence'])

            return filtered_detections

        except Exception as e:
            print(f"Erro durante detecção: {e}")
            return []

    def _process_results(self, result, image_shape: Tuple[int, int, int]) -> List[Dict[str, Any]]:
        detections = []

        if result.boxes is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        class_names = result.names

        for i in range(len(boxes)):
            bbox = boxes[i].tolist()
            confidence = float(confidences[i])
            class_id = int(class_ids[i])
            class_name = class_names[class_id]

            if self._is_shoe_class(class_name):
                detection = {
                    'bbox': bbox,
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name,
                    'center': self._get_bbox_center(bbox),
                    'area': self._calculate_bbox_area(bbox)
                }
                detections.append(detection)

        return detections

    def _is_shoe_class(self, class_name: str) -> bool:
        class_name_lower = class_name.lower()

        if class_name_lower in self.shoe_classes:
            return True

        shoe_keywords = ['shoe', 'sneaker', 'footwear', 'boot', 'sandal', 'tennis', 'athletic']
        for keyword in shoe_keywords:
            if keyword in class_name_lower:
                return True

        return False

    def _filter_shoe_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not detections:
            return []

        filtered = filter_detections_by_size(
            detections, self.min_shoe_size, self.max_shoe_size
        )

        filtered = filter_detections_by_aspect_ratio(
            filtered, self.aspect_ratio_range[0], self.aspect_ratio_range[1]
        )

        filtered = [det for det in filtered if det['confidence'] >= self.confidence_threshold]

        filtered.sort(key=lambda x: x['confidence'], reverse=True)

        return filtered

    def _get_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        return (center_x, center_y)

    def _calculate_bbox_area(self, bbox: List[float]) -> float:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width * height

    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]],
                        show_confidence: bool = True, show_tracking: bool = False) -> np.ndarray:
        result_image = image.copy()

        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']

            if confidence >= 0.8:
                color = COLORS['shoe_detected']
            elif confidence >= 0.6:
                color = COLORS['shoe_tracking']
            else:
                color = (128, 128, 128)

            label = f"{class_name}"
            if show_confidence:
                label += f" {confidence:.2f}"

            if show_tracking and 'id' in detection:
                label += f" ID:{detection['id']}"

            result_image = draw_bbox(
                result_image, bbox, label, confidence, color, 2
            )

        return result_image

    def get_detection_statistics(self) -> Dict[str, Any]:
        if not self.confidence_scores:
            return {
                'total_detections': 0,
                'avg_confidence': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0,
                'avg_detection_time': 0.0
            }

        return {
            'total_detections': self.total_detections,
            'avg_confidence': np.mean(self.confidence_scores),
            'min_confidence': np.min(self.confidence_scores),
            'min_confidence': np.min(self.confidence_scores),
            'max_confidence': np.max(self.confidence_scores),
            'avg_detection_time': np.mean(self.detection_times) if self.detection_times else 0.0,
            'total_confidence_scores': len(self.confidence_scores)
        }

    def reset_statistics(self) -> None:
        self.total_detections = 0
        self.confidence_scores.clear()
        self.detection_times.clear()

    def update_model_config(self, new_config: Dict[str, Any]) -> bool:
        try:
            for key, value in new_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                elif key in self.config:
                    self.config[key] = value

            if 'model_size' in new_config:
                self._load_model()

            print("Configurações do modelo atualizadas com sucesso")
            return True

        except Exception as e:
            print(f"Erro ao atualizar configurações: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        if self.model is None:
            return {'status': 'Modelo não carregado'}

        return {
            'model_name': f"yolov8{self.model_size}",
            'device': str(self.model.device),
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'max_detections': self.max_det,
            'shoe_classes': self.shoe_classes,
            'min_shoe_size': self.min_shoe_size,
            'max_shoe_size': self.max_shoe_size
        }

    def save_model_config(self, filepath: str) -> bool:
        try:
            import json

            config_data = {
                'model_config': self.config,
                'shoe_config': self.shoe_config,
                'statistics': self.get_detection_statistics(),
                'model_info': self.get_model_info(),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            print(f"Configurações salvas em: {filepath}")
            return True

        except Exception as e:
            print(f"Erro ao salvar configurações: {e}")
            return False
