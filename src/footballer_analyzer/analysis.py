"""编排素材分析：核实视频末尾、人物追踪，再自动标定或等待手动固定标定。"""

from dataclasses import replace

import cv2

from .pipeline import track
from .video import inspect_video
from .white_line_pipeline import calibrate_run


def analyze_clip(config, progress):
    """生成人物缓存，按配置自动标定或等待手动标定；progress 接收阶段、数量及运行路径。

    未指定结束帧时扫描至可解码末尾，避免素材容器虚报帧数；指定区间仍严格校验。
    """
    info = inspect_video(config.video)
    progress("inspecting", 0, info.frame_count, None)
    if config.end_frame is None:
        capture = cv2.VideoCapture(str(config.video))
        decoded = 0
        try:
            while capture.grab():
                decoded += 1
                if decoded % 25 == 0:
                    progress("inspecting", decoded, info.frame_count, None)
        finally:
            capture.release()
        if decoded <= config.start_frame:
            raise ValueError("分析起点之后没有可解码视频帧")
        config = replace(config, end_frame=decoded)
    elif config.end_frame > info.frame_count:
        raise ValueError("分析区间超出视频范围")

    def tracking_progress(stage, processed, total):
        """转发人物预采样与逐帧追踪进度。"""
        progress(stage, processed, total, None)

    run_dir = track(config, tracking_progress)
    total = config.end_frame - config.start_frame
    if config.calibration_method == "manual":
        progress("awaiting_calibration", total, total, run_dir)
        return run_dir
    progress("calibrating", 0, total, run_dir)

    def calibration_progress(processed, expected):
        """转发同一源帧区间的白线标定进度。"""
        progress("calibrating", processed, expected, run_dir)

    calibrate_run(run_dir, config.pitch_length_m, config.pitch_width_m, calibration_progress)
    return run_dir
