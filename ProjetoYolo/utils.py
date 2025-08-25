import cv2
import numpy as np
import json
import os
import time
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def setup_logging(log_level: str = 'INFO', log_file: str = None, console_output: bool = True) -> logging.Logger:
    logger = logging.getLogger('ShoeDetector')
    logger.setLevel(getattr(logging, log_level.upper()))

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def calculate_distance(box1: List[float], box2: List[float]) -> float:
    center1_x = (box1[0] + box1[2]) / 2
    center1_y = (box1[1] + box1[3]) / 2
    center2_x = (box2[0] + box2[2]) / 2
    center2_y = (box2[1] + box2[3]) / 2

    return np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)


def filter_detections_by_size(detections: List[Dict], min_size: int = 20, max_size: int = 300) -> List[Dict]:
    filtered = []
    for det in detections:
        width = det['bbox'][2] - det['bbox'][0]
        height = det['bbox'][3] - det['bbox'][1]
        size = max(width, height)

        if min_size <= size <= max_size:
            filtered.append(det)

    return filtered


def filter_detections_by_aspect_ratio(detections: List[Dict], min_ratio: float = 0.5, max_ratio: float = 2.0) -> List[
    Dict]:
    filtered = []
    for det in detections:
        width = det['bbox'][2] - det['bbox'][0]
        height = det['bbox'][3] - det['bbox'][1]

        if height > 0:
            ratio = width / height
            if min_ratio <= ratio <= max_ratio:
                filtered.append(det)

    return filtered


def draw_bbox(image: np.ndarray, bbox: List[float], label: str = None,
              confidence: float = None, color: Tuple[int, int, int] = (0, 255, 0),
              thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    if label or confidence is not None:
        text = label or ""
        if confidence is not None:
            text += f" {confidence:.2f}"

        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x1
        text_y = y1 - 10 if y1 > 20 else y1 + text_size[1] + 10

        cv2.rectangle(image, (text_x, text_y - text_size[1] - 5),
                      (text_x + text_size[0], text_y + 5), color, -1)

        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

    return image


def draw_roi(image: np.ndarray, roi: List[int], color: Tuple[int, int, int] = (0, 255, 0),
             thickness: int = 2, label: str = "ROI") -> np.ndarray:
    x1, y1, x2, y2 = roi
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, thickness)

    return image


def resize_image(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    h, w = image.shape[:2]
    aspect_ratio = w / h
    target_ratio = target_width / target_height

    if aspect_ratio > target_ratio:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    resized = cv2.resize(image, (new_width, new_height))

    result = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2

    result[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

    return result


def calculate_fps(start_time: float, frame_count: int) -> float:
    elapsed_time = time.time() - start_time
    return frame_count / elapsed_time if elapsed_time > 0 else 0


def save_results(results: Dict[str, Any], output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)


def create_summary_plot(results: Dict[str, Any], output_file: str) -> None:
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Resumo da Detecção de Tênis', fontsize=16, fontweight='bold')

    total_detected = results.get('total_detected', 0)
    total_collected = results.get('total_collected', 0)
    remaining = results.get('remaining', 0)

    labels = ['Detectados', 'Coletados', 'Restantes']
    values = [total_detected, total_collected, remaining]
    colors = ['#2ecc71', '#e74c3c', '#f39c12']

    axes[0, 0].pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Distribuição dos Tênis')

    if 'frame_history' in results:
        frames = [f['frame'] for f in results['frame_history']]
        counts = [f['count'] for f in results['frame_history']]

        axes[0, 1].plot(frames, counts, 'b-', linewidth=2)
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Tênis Detectados')
        axes[0, 1].set_title('Evolução da Contagem')
        axes[0, 1].grid(True, alpha=0.3)

    if 'confidence_scores' in results:
        confidences = results['confidence_scores']
        axes[1, 0].hist(confidences, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Confiança')
        axes[1, 0].set_ylabel('Frequência')
        axes[1, 0].set_title('Distribuição de Confiança')
        axes[1, 0].grid(True, alpha=0.3)

    if 'performance_metrics' in results:
        perf = results['performance_metrics']
        metrics = ['FPS Médio', 'Tempo Total', 'Frames Processados']
        values = [perf.get('avg_fps', 0), perf.get('total_time', 0), perf.get('total_frames', 0)]

        bars = axes[1, 1].bar(metrics, values, color=['#3498db', '#e67e22', '#9b59b6'])
        axes[1, 1].set_title('Métricas de Performance')
        axes[1, 1].tick_params(axis='x', rotation=45)

        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                            f'{value:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}h {remaining_minutes}m {remaining_seconds:.1f}s"


def validate_roi(roi: List[int], image_shape: Tuple[int, int]) -> bool:
    if len(roi) != 4:
        return False

    x1, y1, x2, y2 = roi
    height, width = image_shape

    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return False

    if x2 <= x1 or y2 <= y1:
        return False

    return True


def create_output_directories() -> None:
    directories = ['outputs', 'outputs/videos', 'outputs/images', 'outputs/logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
