MODEL_CONFIG = {
    'model_size': 'n',
    'model_path': 'yolov8n.pt',
    'confidence_threshold': 0.5,
    'iou_threshold': 0.45,
    'max_det': 100,
}

ROI_CONFIG = {
    'margin': 50,
    'auto_detect': True,
    'manual_roi': None,
    'roi_color': (0, 255, 0),
    'roi_thickness': 2,
}

TRACKING_CONFIG = {
    'tracking_threshold': 0.7,
    'disappearance_frames': 5,
    'min_tracking_confidence': 0.3,
    'max_tracking_distance': 100,
}

VIDEO_CONFIG = {
    'frame_width': 640,
    'frame_height': 480,
    'fps_target': 30,
    'save_processed': True,
    'output_format': 'mp4',
}

UI_CONFIG = {
    'show_fps': True,
    'show_confidence': True,
    'show_tracking': True,
    'show_roi': True,
    'font_scale': 0.6,
    'font_thickness': 2,
}

COLORS = {
    'shoe_detected': (0, 255, 0),
    'shoe_collected': (0, 0, 255),
    'shoe_tracking': (255, 0, 0),
    'roi_boundary': (0, 255, 0),
    'text': (255, 255, 255),
    'background': (0, 0, 0),
}

LOGGING_CONFIG = {
    'level': 'INFO',
    'save_to_file': True,
    'log_file': 'outputs/detection.log',
    'console_output': True,
}

OUTPUT_CONFIG = {
    'save_results': True,
    'save_video': True,
    'save_images': False,
    'output_dir': 'outputs',
    'results_file': 'detection_results.json',
}

PERFORMANCE_CONFIG = {
    'use_gpu': True,
    'optimize_memory': True,
    'batch_size': 1,
    'threads': 4,
}

SHOE_DETECTION_CONFIG = {
    'min_shoe_size': 20,
    'max_shoe_size': 300,
    'aspect_ratio_range': (0.5, 2.0),
    'class_names': ['shoe', 'sneaker', 'footwear'],
}

EVENT_CONFIG = {
    'collection_timeout': 2.0,
    'min_collection_confidence': 0.8,
    'save_event_log': True,
    'event_log_file': 'events.log',
}

