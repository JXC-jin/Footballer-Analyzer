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

    def __init__(self, source_fps, stride, track_activation_threshold=0.25, lost_track_buffer_s=3.0, minimum_matching_threshold=0.8):
        """初始化跨帧状态；丢失缓冲以秒配置，再换算成实际采样帧数。

        检测下限应为 0.1，使低分框能参与第二次关联；新轨迹需达到激活阈值加 0.1。
        库的缓冲按 30 FPS 归一化，因此传入已换算的帧数并固定归一化帧率。
        """
        if not math.isfinite(source_fps) or source_fps <= 0 or type(stride) is not int or stride <= 0:
            raise ValueError("源帧率和采样间隔必须为正")
        if not math.isfinite(lost_track_buffer_s) or lost_track_buffer_s <= 0:
            raise ValueError("丢失缓冲必须为有限正秒数")
        if not 0.1 < track_activation_threshold < 0.9 or not 0 < minimum_matching_threshold < 1:
            raise ValueError("跟踪激活或匹配阈值无效")
        self.tracker = sv.ByteTrack(
            frame_rate=30,
            lost_track_buffer=max(1, math.ceil(source_fps / stride * lost_track_buffer_s)),
            minimum_consecutive_frames=1,
            track_activation_threshold=track_activation_threshold,
            minimum_matching_threshold=minimum_matching_threshold,
        )
        self.previous_timestamp = -math.inf

    def update(self, detections, timestamp_s):
        """关联当前帧并返回带 ID 的检测；时间不递增时抛出 ValueError。"""
        if not math.isfinite(timestamp_s) or timestamp_s <= self.previous_timestamp:
            raise ValueError("跟踪时间戳必须递增")
        self.previous_timestamp = timestamp_s
        return self.tracker.update_with_detections(detections)
