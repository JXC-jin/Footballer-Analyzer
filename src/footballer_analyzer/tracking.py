"""封装 ByteTrack 跨帧记忆；按实际采样帧率设置缓冲，ID 仅在本次运行内有效。"""

from typing import Protocol
import math

import supervision as sv


class Tracker(Protocol):
    """跟踪接口：按时间顺序关联检测，为匹配观测附加轨迹 ID。"""

    def update(self, detections: sv.Detections, timestamp_s: float) -> sv.Detections:
        """接收当前帧检测和源时间（秒），返回带 tracker_id 的观测。"""
        ...


class ByteTracker:
    """维护同一视频窗口的轨迹状态，通过运动预测和框匹配延续编号。"""

    def __init__(self, source_fps, stride):
        """用源帧率与采样间隔初始化跟踪器，状态在后续 update 调用间保留。"""
        self.tracker = sv.ByteTrack(
            frame_rate=max(1, round(source_fps / stride)),
            minimum_consecutive_frames=1,
            track_activation_threshold=0.25,
        )
        self.previous_timestamp = -math.inf

    def update(self, detections, timestamp_s):
        """关联当前帧并返回带 ID 的检测；时间不递增时抛出 ValueError。"""
        if timestamp_s <= self.previous_timestamp:
            raise ValueError("跟踪时间戳必须递增")
        self.previous_timestamp = timestamp_s
        return self.tracker.update_with_detections(detections)
