"""用 YOLO 检测单帧人物，输出原图像素框；跨帧身份由 tracking 模块维护。"""

from typing import Protocol

import numpy as np
import supervision as sv
from ultralytics import YOLO


class Detector(Protocol):
    """检测接口：接收 BGR 图像，返回 supervision 检测结果。"""

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """返回当前帧的像素框、类别和置信度，不分配轨迹 ID。"""
        ...


class YoloDetector:
    """CPU 上的 YOLO person 检测器，仅保留 COCO 类别 0。"""

    def __init__(self, weights, imgsz=1280, confidence=0.1, nms_iou=0.5):
        """加载本地权重；confidence 为检测下限，nms_iou 控制重复框去重。"""
        if not weights.is_file():
            raise ValueError(f"缺少本地模型权重：{weights}")
        self.model = YOLO(str(weights))
        self.imgsz = imgsz
        self.confidence = confidence
        self.nms_iou = nms_iou

    def detect(self, frame):
        """检测 BGR 帧中的人物，返回原图坐标下的 sv.Detections。"""
        result = self.model.predict(frame, imgsz=self.imgsz, conf=self.confidence, iou=self.nms_iou, classes=[0], device="cpu", verbose=False)[0]
        return sv.Detections.from_ultralytics(result)
