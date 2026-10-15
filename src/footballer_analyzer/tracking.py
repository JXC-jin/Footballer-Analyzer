from typing import Protocol
import math

import supervision as sv


class Tracker(Protocol):
    def update(self, detections: sv.Detections, timestamp_s: float) -> sv.Detections: ...


class ByteTracker:
    def __init__(self, source_fps, stride):
        self.tracker = sv.ByteTrack(
            frame_rate=max(1, round(source_fps / stride)),
            minimum_consecutive_frames=1,
            track_activation_threshold=0.25,
        )
        self.previous_timestamp = -math.inf

    def update(self, detections, timestamp_s):
        if timestamp_s <= self.previous_timestamp:
            raise ValueError("跟踪时间戳必须递增")
        self.previous_timestamp = timestamp_s
        return self.tracker.update_with_detections(detections)
