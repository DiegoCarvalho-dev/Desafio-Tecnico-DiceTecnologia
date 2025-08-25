import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, deque
from config import TRACKING_CONFIG, EVENT_CONFIG


class EventTracker:
    def __init__(self, config: Dict[str, Any] = None):

        self.config = config or TRACKING_CONFIG
        self.tracked_objects = {}
        self.object_history = defaultdict(deque)
        self.collection_events = []
        self.disappearance_frames = defaultdict(int)
        self.next_id = 1

        self.tracking_threshold = self.config.get('tracking_threshold', 0.7)
        self.disappearance_frames_limit = self.config.get('disappearance_frames', 5)
        self.max_tracking_distance = self.config.get('max_tracking_distance', 100)
        self.min_tracking_confidence = self.config.get('min_tracking_confidence', 0.3)

        self.total_detected = 0
        self.total_collected = 0
        self.current_count = 0
        self.max_concurrent = 0

    def update_tracking(self, detections: List[Dict], frame_number: int) -> Dict[str, Any]:
        current_objects = {}
        new_detections = []
        updated_objects = []

        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']

            if confidence < self.min_tracking_confidence:
                continue

            best_match_id = self._find_best_match(bbox, confidence)

            if best_match_id is not None:
                self._update_object(best_match_id, bbox, confidence, frame_number)
                current_objects[best_match_id] = self.tracked_objects[best_match_id]
                updated_objects.append(best_match_id)
            else:
                new_id = self._create_new_object(bbox, confidence, frame_number)
                current_objects[new_id] = self.tracked_objects[new_id]
                new_detections.append(new_id)

        disappeared_objects = self._check_disappearances(frame_number)

        self._update_statistics(current_objects, new_detections, disappeared_objects)

        return {
            'current_objects': current_objects,
            'new_detections': new_detections,
            'updated_objects': updated_objects,
            'disappeared_objects': disappeared_objects,
            'total_count': len(current_objects),
            'new_count': len(new_detections),
            'collected_count': len(disappeared_objects)
        }

    def _find_best_match(self, bbox: List[float], confidence: float) -> Optional[int]:
        best_match_id = None
        best_score = 0

        for obj_id, obj_info in self.tracked_objects.items():
            if obj_info.get('collected', False):
                continue

            score = self._calculate_matching_score(bbox, obj_info['bbox'], confidence)

            if score > best_score and score >= self.tracking_threshold:
                best_score = score
                best_match_id = obj_id

        return best_match_id

    def _calculate_matching_score(self, bbox1: List[float], bbox2: List[float],
                                  confidence: float) -> float:

        iou = self._calculate_iou(bbox1, bbox2)


        center1 = self._get_bbox_center(bbox1)
        center2 = self._get_bbox_center(bbox2)
        distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

        max_distance = self.max_tracking_distance
        distance_score = max(0, 1 - distance / max_distance)

        final_score = 0.5 * iou + 0.3 * distance_score + 0.2 * confidence

        return final_score

    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _get_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:

        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        return (center_x, center_y)

    def _create_new_object(self, bbox: List[float], confidence: float,
                           frame_number: int) -> int:

        obj_id = self.next_id
        self.next_id += 1

        obj_info = {
            'id': obj_id,
            'bbox': bbox,
            'confidence': confidence,
            'first_seen': frame_number,
            'last_seen': frame_number,
            'collected': False,
            'collection_frame': None,
            'total_frames': 1,
            'avg_confidence': confidence,
            'movement_history': [bbox]
        }

        self.tracked_objects[obj_id] = obj_info
        self.object_history[obj_id].append({
            'frame': frame_number,
            'bbox': bbox,
            'confidence': confidence
        })

        return obj_id

    def _update_object(self, obj_id: int, bbox: List[float], confidence: float,
                       frame_number: int) -> None:
        obj_info = self.tracked_objects[obj_id]

        obj_info['bbox'] = bbox
        obj_info['confidence'] = confidence
        obj_info['last_seen'] = frame_number
        obj_info['total_frames'] += 1

        total_confidence = obj_info['avg_confidence'] * (obj_info['total_frames'] - 1) + confidence
        obj_info['avg_confidence'] = total_confidence / obj_info['total_frames']

        obj_info['movement_history'].append(bbox)
        if len(obj_info['movement_history']) > 10:
            obj_info['movement_history'].pop(0)

        self.object_history[obj_id].append({
            'frame': frame_number,
            'bbox': bbox,
            'confidence': confidence
        })

        if len(self.object_history[obj_id]) > 50:
            self.object_history[obj_id].popleft()

        self.disappearance_frames[obj_id] = 0

    def _check_disappearances(self, frame_number: int) -> List[int]:
        collected_objects = []

        for obj_id, obj_info in self.tracked_objects.items():
            if obj_info.get('collected', False):
                continue

            frames_since_last_seen = frame_number - obj_info['last_seen']
            self.disappearance_frames[obj_id] = frames_since_last_seen

            if frames_since_last_seen >= self.disappearance_frames_limit:
                obj_info['collected'] = True
                obj_info['collection_frame'] = frame_number
                collected_objects.append(obj_id)

                self._record_collection_event(obj_id, obj_info, frame_number)

        return collected_objects

    def _record_collection_event(self, obj_id: int, obj_info: Dict, frame_number: int) -> None:
        event = {
            'timestamp': time.time(),
            'frame_number': frame_number,
            'object_id': obj_id,
            'bbox': obj_info['bbox'],
            'confidence': obj_info['confidence'],
            'total_frames_tracked': obj_info['total_frames'],
            'avg_confidence': obj_info['avg_confidence'],
            'first_seen_frame': obj_info['first_seen'],
            'last_seen_frame': obj_info['last_seen']
        }

        self.collection_events.append(event)

    def _update_statistics(self, current_objects: Dict, new_detections: List[int],
                           disappeared_objects: List[int]) -> None:

        self.current_count = len(current_objects)
        self.total_detected += len(new_detections)
        self.total_collected += len(disappeared_objects)

        if self.current_count > self.max_concurrent:
            self.max_concurrent = self.current_count

    def get_tracking_summary(self) -> Dict[str, Any]:
        active_objects = [obj for obj in self.tracked_objects.values() if not obj.get('collected', False)]

        return {
            'total_detected': self.total_detected,
            'total_collected': self.total_collected,
            'current_count': self.current_count,
            'max_concurrent': self.max_concurrent,
            'active_objects': len(active_objects),
            'total_tracked': len(self.tracked_objects),
            'collection_events': len(self.collection_events)
        }

    def get_object_info(self, obj_id: int) -> Optional[Dict[str, Any]]:
        return self.tracked_objects.get(obj_id)

    def get_collection_events(self) -> List[Dict[str, Any]]:
        return self.collection_events.copy()

    def get_object_movement(self, obj_id: int) -> List[Dict[str, Any]]:
        return list(self.object_history.get(obj_id, []))

    def reset_tracking(self) -> None:
        self.tracked_objects.clear()
        self.object_history.clear()
        self.collection_events.clear()
        self.disappearance_frames.clear()
        self.next_id = 1
        self.total_detected = 0
        self.total_collected = 0
        self.current_count = 0
        self.max_concurrent = 0

    def export_tracking_data(self) -> Dict[str, Any]:

        return {
            'tracking_summary': self.get_tracking_summary(),
            'tracked_objects': self.tracked_objects,
            'collection_events': self.collection_events,
            'object_histories': dict(self.object_history),
            'configuration': self.config
        }

