"""建立图像到米制场地的单应性；校验对应点、可靠区域、适用窗口和确认状态。"""

from pathlib import Path
import math

import cv2
import numpy as np

from .storage import read_json, read_yaml, write_yaml


def build_calibration(manifest, payload):
    """根据运行清单和人工对应点构造标定字典，非法几何或区间抛出 ValueError。

    图像点使用源像素，场地点与尺寸使用米；至少四对非共线点，默认可靠区域为凸包。
    """
    image_points = np.asarray(payload.get("image_points", []), dtype=float)
    pitch_points = np.asarray(payload.get("pitch_points", []), dtype=float)
    if image_points.ndim != 2 or image_points.shape[1:] != (2,) or len(image_points) < 4 or image_points.shape != pitch_points.shape:
        raise ValueError("至少需要 4 对图像点与球场坐标")
    if not np.isfinite(image_points).all() or not np.isfinite(pitch_points).all():
        raise ValueError("标定点必须是有限坐标")
    for points in [image_points, pitch_points]:
        if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
            raise ValueError("标定点共线，无法求单应性")
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError("标定点不能重复")
    video = manifest["video"]
    if np.any(image_points < 0) or np.any(image_points[:, 0] >= video["width"]) or np.any(image_points[:, 1] >= video["height"]):
        raise ValueError("图像标定点超出视频范围")
    length = float(payload.get("pitch_length_m", manifest["config"]["pitch_length_m"]))
    width = float(payload.get("pitch_width_m", manifest["config"]["pitch_width_m"]))
    if not math.isfinite(length) or not math.isfinite(width) or min(length, width) <= 0:
        raise ValueError("球场尺寸无效")
    if np.any(pitch_points < 0) or np.any(pitch_points[:, 0] > length) or np.any(pitch_points[:, 1] > width):
        raise ValueError("场地点超出设置的球场尺寸")
    homography, inliers = cv2.findHomography(image_points, pitch_points, cv2.RANSAC, 1.0)
    if homography is None or not np.isfinite(homography).all() or np.linalg.matrix_rank(homography) < 3:
        raise ValueError("无法求得有效单应性，请检查对应点")
    start = payload.get("start_frame", manifest["start_frame"])
    end = payload.get("end_frame", manifest["end_frame"])
    reference = payload.get("reference_frame", start)
    if any(type(value) is not int for value in [start, end, reference]) or not manifest["start_frame"] <= start <= reference < end <= manifest["end_frame"]:
        raise ValueError("标定有效帧区间或参考帧不匹配")
    roi = np.asarray(payload.get("roi", cv2.convexHull(image_points.astype(np.float32)).reshape(-1, 2).tolist()), dtype=float)
    if roi.ndim != 2 or roi.shape[1:] != (2,) or len(roi) < 3 or not np.isfinite(roi).all() or cv2.contourArea(roi.astype(np.float32)) < 1:
        raise ValueError("可靠区域必须为至少三个点构成的有效多边形")
    if np.any(roi < 0) or np.any(roi[:, 0] >= video["width"]) or np.any(roi[:, 1] >= video["height"]):
        raise ValueError("可靠区域超出视频范围")
    projected = cv2.perspectiveTransform(image_points.reshape(-1, 1, 2), homography).reshape(-1, 2)
    return {
        "schema_version": 1, "video_sha256": manifest["video_sha256"],
        "width": video["width"], "height": video["height"],
        "reference_frame": reference, "start_frame": start, "end_frame": end,
        "pitch_length_m": length, "pitch_width_m": width,
        "image_points": image_points.tolist(), "pitch_points": pitch_points.tolist(),
        "homography": homography.tolist(), "roi": roi.tolist(),
        "fit_rmse_m": float(np.sqrt(np.mean(np.sum((projected - pitch_points) ** 2, axis=1)))),
        "inlier_count": int(inliers.sum()), "reviewed": payload.get("reviewed") is True,
        "note": str(payload.get("note", "")),
    }


def validate_calibration(calibration, manifest, require_reviewed=True):
    """检查标定是否匹配视频摘要与分辨率，并按需要求人工确认；失败时报错。"""
    if calibration is None:
        raise ValueError("请先在调试台完成场地标定")
    if calibration["video_sha256"] != manifest["video_sha256"] or calibration["width"] != manifest["video"]["width"] or calibration["height"] != manifest["video"]["height"]:
        raise ValueError("标定与源视频不匹配")
    if require_reviewed and calibration.get("reviewed") is not True:
        raise ValueError("标定为草稿：请检查首、中、末帧后确认，再生成统计报告")


def save_calibration(run_dir: Path, payload):
    """构建标定，写入运行目录的 calibration.yaml，并返回标定字典。"""
    calibration = build_calibration(read_json(run_dir / "manifest.json"), payload)
    write_yaml(run_dir / "calibration.yaml", calibration)
    return calibration


def load_calibration(run_dir):
    """读取运行目录的标定字典；尚无标定文件时返回 None。"""
    return read_yaml(run_dir / "calibration.yaml")


def project_observation(row, calibration):
    """投影框底边中点，返回 (x 米, y 米, 无效原因)。

    检查有效帧区间、脚点裁切、可靠区域和场地边界；无法投影时坐标为 None。
    """
    if calibration is None:
        return None, None, "uncalibrated"
    if not calibration["start_frame"] <= row["frame_index"] < calibration["end_frame"]:
        return None, None, "outside_calibration_window"
    foot_x = (row["bbox_x1"] + row["bbox_x2"]) / 2
    foot_y = row["bbox_y2"]
    transformed = np.asarray(calibration["homography"]) @ np.array([foot_x, foot_y, 1.0])
    if not np.isfinite(transformed).all() or abs(transformed[2]) < 1e-8:
        return None, None, "invalid_projection"
    world_x, world_y = (transformed[:2] / transformed[2]).tolist()
    if row["bbox_y2"] >= calibration["height"] - 2:
        return world_x, world_y, "clipped_feet"
    if cv2.pointPolygonTest(np.asarray(calibration["roi"], np.float32), (foot_x, foot_y), False) < 0:
        return world_x, world_y, "outside_roi"
    if not 0 <= world_x <= calibration["pitch_length_m"] or not 0 <= world_y <= calibration["pitch_width_m"]:
        return world_x, world_y, "outside_pitch"
    return world_x, world_y, None
