"""解码恒定帧率视频；抽帧保留源帧号与源时间，解码失败时显式报错。"""

from dataclasses import asdict, dataclass
from pathlib import Path
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    """源视频元数据：像素尺寸、每秒帧数与总帧数。"""

    width: int
    height: int
    fps: float
    frame_count: int

    def snapshot(self):
        """返回可写入运行清单的视频元数据字典。"""
        return asdict(self)


@dataclass
class FramePacket:
    """一帧观测：源帧号、源时间（秒）和 OpenCV BGR 图像。"""

    frame_index: int
    timestamp_s: float
    image: np.ndarray


def inspect_video(path: Path) -> VideoInfo:
    """返回视频元数据；无法打开或尺寸、帧率、帧数无效时抛出 ValueError。"""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"无法打开视频：{path}")
        info = VideoInfo(
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(capture.get(cv2.CAP_PROP_FPS)),
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        if not math.isfinite(info.fps) or info.fps <= 0 or min(info.width, info.height, info.frame_count) <= 0:
            raise ValueError("视频尺寸、帧率或帧数无效；首版要求恒定帧率素材")
        return info
    finally:
        capture.release()


def read_frames(path: Path, start_frame=0, end_frame=None, stride=1):
    """按 stride 迭代 [start_frame, end_frame) 内的 FramePacket。

    end_frame 为空时读取到末尾；时间为源帧号除以源帧率，越界或解码失败时报错。
    """
    info = inspect_video(path)
    stop_frame = info.frame_count if end_frame is None else end_frame
    if stride < 1 or start_frame < 0 or stop_frame > info.frame_count or start_frame >= stop_frame:
        raise ValueError("视频分析帧区间或 stride 无效")
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_index in range(start_frame, stop_frame):
            if (frame_index - start_frame) % stride:
                if not capture.grab():
                    raise ValueError(f"视频在第 {frame_index} 帧提前结束")
                continue
            success, frame = capture.read()
            if not success:
                raise ValueError(f"无法解码第 {frame_index} 帧")
            yield FramePacket(frame_index, frame_index / info.fps, frame)
    finally:
        capture.release()


def read_frame(path: Path, frame_index: int):
    """读取指定源帧的 BGR 图像；沿用 read_frames 的区间与解码校验。"""
    return next(read_frames(path, frame_index, frame_index + 1)).image
