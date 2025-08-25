import unittest
import numpy as np
import cv2
import os
import tempfile
import json
from unittest.mock import Mock, patch


from shoe_detector import ShoeDetector
from roi_manager import ROIManager
from event_tracker import EventTracker
from utils import (
    calculate_iou, calculate_distance, filter_detections_by_size,
    filter_detections_by_aspect_ratio, draw_bbox, validate_roi
)


class TestShoeDetector(unittest.TestCase):
    def setUp(self):
        self.mock_model = Mock()
        self.mock_model.device.type = "cpu"

        with patch('ultralytics.YOLO', return_value=self.mock_model):
            self.detector = ShoeDetector()
            self.detector.model = self.mock_model

    def test_initialization(self):
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.confidence_threshold, 0.5)
        self.assertEqual(self.detector.model_size, 'n')

    def test_is_shoe_class(self):
        self.assertTrue(self.detector._is_shoe_class('shoe'))
        self.assertTrue(self.detector._is_shoe_class('sneaker'))
        self.assertTrue(self.detector._is_shoe_class('footwear'))

        self.assertFalse(self.detector._is_shoe_class('person'))
        self.assertFalse(self.detector._is_shoe_class('car'))

    def test_bbox_center_calculation(self):

        bbox = [10, 20, 30, 40]
        center = self.detector._get_bbox_center(bbox)

        self.assertEqual(center, (20, 30))

    def test_bbox_area_calculation(self):
        bbox = [0, 0, 10, 20]
        area = self.detector._calculate_bbox_area(bbox)

        self.assertEqual(area, 200)

    def test_filter_shoe_detections(self):
        detections = [
            {'bbox': [0, 0, 50, 50], 'confidence': 0.8, 'class_name': 'shoe'},
            {'bbox': [100, 100, 200, 200], 'confidence': 0.3, 'class_name': 'shoe'},
            {'bbox': [0, 0, 10, 10], 'confidence': 0.9, 'class_name': 'shoe'}
        ]

        filtered = self.detector._filter_shoe_detections(detections)

        self.assertLessEqual(len(filtered), len(detections))

    def test_detection_statistics(self):
        self.detector.confidence_scores = [0.8, 0.9, 0.7]
        self.detector.detection_times = [0.1, 0.2, 0.15]
        self.detector.total_detections = 3

        stats = self.detector.get_detection_statistics()

        self.assertEqual(stats['total_detections'], 3)
        self.assertAlmostEqual(stats['avg_confidence'], 0.8, places=1)
        self.assertAlmostEqual(stats['avg_detection_time'], 0.15, places=2)


class TestROIManager(unittest.TestCase):
    def setUp(self):
        self.roi_manager = ROIManager()

    def test_initialization(self):
        self.assertIsNotNone(self.roi_manager)
        self.assertIsNone(self.roi_manager.roi)

    def test_auto_detect_roi(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        roi = self.roi_manager.auto_detect_roi(image)

        self.assertEqual(len(roi), 4)
        self.assertGreater(roi[2], roi[0])  # x2 > x1
        self.assertGreater(roi[3], roi[1])  # y2 > y1

    def test_set_roi(self):
        test_roi = [50, 50, 590, 430]

        result = self.roi_manager.set_roi(test_roi)

        self.assertTrue(result)
        self.assertEqual(self.roi_manager.roi, test_roi)

    def test_invalid_roi(self):
        invalid_roi = [100, 100, 50, 50]  # x2 < x1

        result = self.roi_manager.set_roi(invalid_roi)

        self.assertFalse(result)

    def test_point_in_roi(self):
        self.roi_manager.roi = [0, 0, 100, 100]

        self.assertTrue(self.roi_manager.is_point_in_roi((50, 50)))

        self.assertFalse(self.roi_manager.is_point_in_roi((150, 150)))

    def test_bbox_in_roi(self):
        self.roi_manager.roi = [0, 0, 100, 100]

        bbox_inside = [10, 10, 90, 90]
        self.assertTrue(self.roi_manager.is_bbox_in_roi(bbox_inside))

        bbox_partial = [50, 50, 150, 150]
        self.assertTrue(self.roi_manager.is_bbox_in_roi(bbox_partial, threshold=0.1))

        bbox_outside = [150, 150, 200, 200]
        self.assertFalse(self.roi_manager.is_bbox_in_roi(bbox_outside))

    def test_roi_area_calculation(self):
        self.roi_manager.roi = [0, 0, 100, 100]
        area = self.roi_manager.get_roi_area()

        self.assertEqual(area, 10000)

    def test_roi_center(self):
        self.roi_manager.roi = [0, 0, 100, 100]
        center = self.roi_manager.get_roi_center()

        self.assertEqual(center, (50, 50))


class TestEventTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = EventTracker()

    def test_initialization(self):
        self.assertIsNotNone(self.tracker)
        self.assertEqual(self.tracker.total_detected, 0)
        self.assertEqual(self.tracker.total_collected, 0)

    def test_create_new_object(self):
        bbox = [0, 0, 50, 50]
        confidence = 0.8
        frame_number = 1

        obj_id = self.tracker._create_new_object(bbox, confidence, frame_number)

        self.assertEqual(obj_id, 1)
        self.assertIn(obj_id, self.tracker.tracked_objects)

        obj_info = self.tracker.tracked_objects[obj_id]
        self.assertEqual(obj_info['bbox'], bbox)
        self.assertEqual(obj_info['confidence'], confidence)
        self.assertEqual(obj_info['first_seen'], frame_number)

    def test_update_tracking(self):
        detections = [
            {'bbox': [0, 0, 50, 50], 'confidence': 0.8},
            {'bbox': [100, 100, 150, 150], 'confidence': 0.9}
        ]

        frame_number = 1
        result = self.tracker.update_tracking(detections, frame_number)

        self.assertEqual(result['total_count'], 2)
        self.assertEqual(result['new_count'], 2)
        self.assertEqual(result['collected_count'], 0)

    def test_object_disappearance(self):
        bbox = [0, 0, 50, 50]
        confidence = 0.8
        frame_number = 1

        obj_id = self.tracker._create_new_object(bbox, confidence, frame_number)

        current_frame = frame_number + 10

        disappeared = self.tracker._check_disappearances(current_frame)

        self.assertIn(obj_id, disappeared)
        self.assertTrue(self.tracker.tracked_objects[obj_id]['collected'])

    def test_tracking_summary(self):
        bbox = [0, 0, 50, 50]
        confidence = 0.8

        self.tracker._create_new_object(bbox, confidence, 1)
        self.tracker._create_new_object(bbox, confidence, 2)

        summary = self.tracker.get_tracking_summary()

        self.assertEqual(summary['total_detected'], 2)
        self.assertEqual(summary['current_count'], 2)
        self.assertEqual(summary['total_tracked'], 2)


class TestUtils(unittest.TestCase):
    def test_calculate_iou(self):
        box1 = [0, 0, 100, 100]
        box2 = [50, 50, 150, 150]

        iou = calculate_iou(box1, box2)

        self.assertGreaterEqual(iou, 0)
        self.assertLessEqual(iou, 1)

        iou_same = calculate_iou(box1, box1)
        self.assertEqual(iou_same, 1.0)

    def test_calculate_distance(self):
        box1 = [0, 0, 100, 100]
        box2 = [100, 100, 200, 200]

        distance = calculate_distance(box1, box2)

        self.assertGreater(distance, 0)

        expected_distance = np.sqrt(100 ** 2 + 100 ** 2)
        self.assertAlmostEqual(distance, expected_distance, places=1)

    def test_filter_detections_by_size(self):
        detections = [
            {'bbox': [0, 0, 10, 10]},
            {'bbox': [0, 0, 50, 50]},
            {'bbox': [0, 0, 100, 100]},
            {'bbox': [0, 0, 400, 400]}
        ]

        filtered = filter_detections_by_size(detections, min_size=20, max_size=300)

        self.assertEqual(len(filtered), 2)

    def test_filter_detections_by_aspect_ratio(self):
        detections = [
            {'bbox': [0, 0, 50, 100]},
            {'bbox': [0, 0, 100, 100]},
            {'bbox': [0, 0, 200, 100]},
            {'bbox': [0, 0, 300, 100]}
        ]

        filtered = filter_detections_by_aspect_ratio(detections, min_ratio=0.5, max_ratio=2.0)

        self.assertEqual(len(filtered), 3)

    def test_validate_roi(self):
        image_shape = (480, 640)  # (height, width)

        valid_roi = [50, 50, 590, 430]
        self.assertTrue(validate_roi(valid_roi, image_shape))

        invalid_roi = [0, 0, 700, 500]
        self.assertFalse(validate_roi(invalid_roi, image_shape))

        invalid_roi2 = [100, 100, 50, 50]
        self.assertFalse(validate_roi(invalid_roi2, image_shape))


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_model = Mock()
        self.mock_model.device.type = "cpu"

        with patch('ultralytics.YOLO', return_value=self.mock_model):
            self.detector = ShoeDetector()
            self.detector.model = self.mock_model

        self.roi_manager = ROIManager()
        self.event_tracker = EventTracker()

    def test_full_pipeline(self):
        test_roi = [50, 50, 590, 430]
        self.roi_manager.set_roi(test_roi)

        image = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_detections = [
            {'bbox': [100, 100, 150, 150], 'confidence': 0.8, 'class_name': 'shoe'},
            {'bbox': [200, 200, 250, 250], 'confidence': 0.9, 'class_name': 'shoe'}
        ]

        with patch.object(self.detector, 'detect_shoes', return_value=mock_detections):
            detections = self.detector.detect_shoes(image)

            roi_detections = []
            for detection in detections:
                if self.roi_manager.is_bbox_in_roi(detection['bbox']):
                    roi_detections.append(detection)

            tracking_info = self.event_tracker.update_tracking(roi_detections, 1)

            self.assertEqual(len(roi_detections), 2)
            self.assertEqual(tracking_info['total_count'], 2)
            self.assertEqual(tracking_info['new_count'], 2)


def run_tests():
    test_suite = unittest.TestSuite()

    test_classes = [
        TestShoeDetector,
        TestROIManager,
        TestEventTracker,
        TestUtils,
        TestIntegration
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    print("Executando testes do sistema de detecção de tênis...")
    print("=" * 50)

    success = run_tests()

    print("=" * 50)
    if success:
        print(" Todos os testes passaram!")
    else:
        print(" Alguns testes falharam!")

    print(f"\nPara executar testes específicos:")
    print("python -m unittest test_system.TestShoeDetector")
    print("python -m unittest test_system.TestROIManager")
    print("python -m unittest test_system.TestEventTracker")

