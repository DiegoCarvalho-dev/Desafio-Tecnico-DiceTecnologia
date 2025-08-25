import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Optional, Dict, Any
from config import ROI_CONFIG, COLORS
from utils import validate_roi, save_results


class ROIManager:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or ROI_CONFIG
        self.roi = None
        self.roi_config_file = 'outputs/roi_config.json'
        self.is_configuring = False
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.load_roi_config()

    def load_roi_config(self) -> None:
        if os.path.exists(self.roi_config_file):
            try:
                with open(self.roi_config_file, 'r') as f:
                    saved_config = json.load(f)
                    if 'roi' in saved_config and saved_config['roi']:
                        self.roi = saved_config['roi']
                        print(f"ROI carregada: {self.roi}")
            except Exception as e:
                print(f"Erro ao carregar configuração da ROI: {e}")

        if self.roi is None:
            self.roi = self.config.get('manual_roi')

    def save_roi_config(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.roi_config_file), exist_ok=True)
            config_data = {
                'roi': self.roi,
                'config': self.config,
                'timestamp': str(np.datetime64('now'))
            }
            save_results(config_data, self.roi_config_file)
            print(f"Configuração da ROI salva em: {self.roi_config_file}")
        except Exception as e:
            print(f"Erro ao salvar configuração da ROI: {e}")

    def auto_detect_roi(self, image: np.ndarray) -> List[int]:
        if not self.config.get('auto_detect', True):
            return self.roi or [0, 0, image.shape[1], image.shape[0]]

        height, width = image.shape[:2]
        margin = self.config.get('margin', 50)

        x1 = margin
        y1 = margin
        x2 = width - margin
        y2 = height - margin

        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, width, height

        return [x1, y1, x2, y2]

    def set_roi(self, roi: List[int]) -> bool:
        if len(roi) != 4:
            print("ROI deve ter 4 coordenadas: [x1, y1, x2, y2]")
            return False

        self.roi = roi
        self.save_roi_config()
        print(f"ROI definida: {roi}")
        return True

    def get_roi(self, image: np.ndarray = None) -> List[int]:
        if self.roi is None:
            if image is not None:
                self.roi = self.auto_detect_roi(image)
            else:
                self.roi = [50, 50, 590, 430]

        return self.roi

    def is_point_in_roi(self, point: Tuple[int, int]) -> bool:
        if self.roi is None:
            return True

        x, y = point
        x1, y1, x2, y2 = self.roi

        return x1 <= x <= x2 and y1 <= y <= y2

    def is_bbox_in_roi(self, bbox: List[float], threshold: float = 0.5) -> bool:
        if self.roi is None:
            return True

        x1, y1, x2, y2 = bbox
        roi_x1, roi_y1, roi_x2, roi_y2 = self.roi

        intersect_x1 = max(x1, roi_x1)
        intersect_y1 = max(y1, roi_y1)
        intersect_x2 = min(x2, roi_x2)
        intersect_y2 = min(y2, roi_y2)

        if intersect_x2 <= intersect_x1 or intersect_y2 <= intersect_y1:
            return False

        intersection_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
        bbox_area = (x2 - x1) * (y2 - y1)

        if bbox_area == 0:
            return False

        overlap_ratio = intersection_area / bbox_area
        return overlap_ratio >= threshold

    def draw_roi(self, image: np.ndarray) -> np.ndarray:
        if self.roi is None:
            return image

        roi = self.get_roi(image)
        color = self.config.get('roi_color', COLORS['roi_boundary'])
        thickness = self.config.get('roi_thickness', 2)

        cv2.rectangle(image, (roi[0], roi[1]), (roi[2], roi[3]), color, thickness)

        label = "ROI - Chão"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        text_x = roi[0]
        text_y = roi[1] - 10 if roi[1] > text_height + 10 else roi[1] + text_height + 10

        cv2.rectangle(image, (text_x, text_y - text_height - 5),
                      (text_x + text_width, text_y + 5), color, -1)

        cv2.putText(image, label, (text_x, text_y), font, font_scale,
                    (255, 255, 255), font_thickness)

        return image

    def start_manual_config(self) -> None:
        self.is_configuring = True
        self.drawing = False
        self.start_point = None
        self.end_point = None
        print("Configuração manual da ROI iniciada. Clique e arraste para definir a área.")

    def handle_mouse_event(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        if not self.is_configuring:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            print(f"Ponto inicial: ({x}, {y})")

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point and self.end_point:
                x1 = min(self.start_point[0], self.end_point[0])
                y1 = min(self.start_point[1], self.end_point[1])
                x2 = max(self.start_point[0], self.end_point[0])
                y2 = max(self.start_point[1], self.end_point[1])

                self.roi = [x1, y1, x2, y2]
                self.save_roi_config()
                print(f"ROI definida manualmente: {self.roi}")
                self.is_configuring = False

    def draw_preview_roi(self, image: np.ndarray) -> np.ndarray:
        if not self.is_configuring:
            return image

        if self.start_point and self.end_point:
            color = self.config.get('roi_color', COLORS['roi_boundary'])
            thickness = self.config.get('roi_thickness', 2)

            x1 = min(self.start_point[0], self.end_point[0])
            y1 = min(self.start_point[1], self.end_point[1])
            x2 = max(self.start_point[0], self.end_point[0])
            y2 = max(self.start_point[1], self.end_point[1])

            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            instructions = "Clique e arraste para definir ROI. Pressione 'r' para resetar, 'c' para confirmar."
            cv2.putText(image, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

        return image

    def reset_roi(self) -> None:
        self.roi = None
        self.is_configuring = False
        self.drawing = False
        self.start_point = None
        self.end_point = None
        print("ROI resetada para detecção automática")

    def get_roi_area(self) -> float:

        if self.roi is None:
            return 0.0

        width = self.roi[2] - self.roi[0]
        height = self.roi[3] - self.roi[1]
        return width * height

    def get_roi_center(self) -> Tuple[int, int]:

        if self.roi is None:
            return (0, 0)

        center_x = (self.roi[0] + self.roi[2]) // 2
        center_y = (self.roi[1] + self.roi[3]) // 2
        return (center_x, center_y)

    def is_valid(self) -> bool:
        if self.roi is None:
            return False

        return len(self.roi) == 4 and self.roi[2] > self.roi[0] and self.roi[3] > self.roi[1]

