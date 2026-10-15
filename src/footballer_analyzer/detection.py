from typing import Protocol

import numpy as np
import supervision as sv
from ultralytics import YOLO


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> sv.Detections: ...


class YoloDetector:
    def __init__(self, weights, imgsz=640, confidence=0.25):
        if not weights.is_file():
            raise ValueError(f"缺少本地模型权重：{weights}")
        self.model = YOLO(str(weights))
        self.imgsz = imgsz
        self.confidence = confidence

    def detect(self, frame):
        result = self.model.predict(frame, imgsz=self.imgsz, conf=self.confidence, classes=[0], device="cpu", verbose=False)[0]
        return sv.Detections.from_ultralytics(result)
