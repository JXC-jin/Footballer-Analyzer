"""用 YOLO 检测单帧人物与足球，跨帧身份仅由 tracking 模块维护。"""

from typing import Protocol

import numpy as np
import supervision as sv
from ultralytics import YOLO


PERSON_CLASS_ID = 0
SPORTS_BALL_CLASS_ID = 32


class Detector(Protocol):
    """检测接口：接收 BGR 图像，返回人物检测结果。"""

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """返回当前帧的人物像素框、类别和置信度，不分配轨迹 ID。"""
        ...


class YoloDetector:
    """CPU YOLO 检测器；向后兼容人物接口，另提供人物与球单次推理。"""

    def __init__(self, weights, imgsz=1280, confidence=0.1, nms_iou=0.5, ball_detection_enabled=True, ball_confidence=0.15):
        """加载本地权重；球检测阈值与人物检测阈值独立配置。"""
        if not weights.is_file():
            raise ValueError(f"缺少本地模型权重：{weights}")
        self.model = YOLO(str(weights))
        self.imgsz = imgsz
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.ball_detection_enabled = ball_detection_enabled
        self.ball_confidence = ball_confidence

    @staticmethod
    def _empty_detections():
        """返回具备标准空数组结构的 supervision 检测对象。"""
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=float),
            confidence=np.empty((0,), dtype=float),
            class_id=np.empty((0,), dtype=int),
        )

    def _validate_class_names(self, include_ball):
        """拒绝将非 COCO person/sports ball 类别误当作相应目标。"""
        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            indexed_names = {}
            for key, value in names.items():
                try:
                    indexed_names[int(key)] = value
                except (TypeError, ValueError):
                    continue
        elif isinstance(names, (list, tuple)):
            indexed_names = dict(enumerate(names))
        else:
            raise ValueError("模型未提供可验证的类别名称；请使用 COCO person/sports ball 类别权重")

        expected = [(PERSON_CLASS_ID, "person")]
        if include_ball:
            expected.append((SPORTS_BALL_CLASS_ID, "sports ball"))
        for class_id, expected_name in expected:
            actual_name = indexed_names.get(class_id)
            if not isinstance(actual_name, str) or " ".join(actual_name.casefold().split()) != expected_name:
                raise ValueError(
                    f"模型类别不兼容：class {class_id} 必须为 {expected_name!r}，实际为 {actual_name!r}"
                )

    def detect_objects(self, frame):
        """单次 YOLO 推理返回 (人物, 足球)；足球关闭时返回空球检测。"""
        self._validate_class_names(self.ball_detection_enabled)
        classes = [PERSON_CLASS_ID, SPORTS_BALL_CLASS_ID] if self.ball_detection_enabled else [PERSON_CLASS_ID]
        # 推理阈值取两类阈值的较小值；再分别过滤，避免球阈值影响球员候选。
        inference_confidence = min(self.confidence, self.ball_confidence) if self.ball_detection_enabled else self.confidence
        result = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=inference_confidence,
            iou=self.nms_iou,
            classes=classes,
            device="cpu",
            verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        if detections.class_id is None:
            return self._empty_detections(), self._empty_detections()
        class_ids = np.asarray(detections.class_id)
        confidence = np.asarray(detections.confidence)
        people = detections[(class_ids == PERSON_CLASS_ID) & (confidence >= self.confidence)]
        if self.ball_detection_enabled:
            balls = detections[(class_ids == SPORTS_BALL_CLASS_ID) & (confidence >= self.ball_confidence)]
        else:
            balls = self._empty_detections()
        return people, balls

    def detect(self, frame):
        """检测并只返回人物框，维持原有 detect(frame) 调用语义。"""
        self._validate_class_names(include_ball=False)
        result = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.confidence,
            iou=self.nms_iou,
            classes=[PERSON_CLASS_ID],
            device="cpu",
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(result)
