"""复查静态场地标定：导出首中末帧对拍图、投影数量与独立参考点像素误差。"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from footballer_analyzer.calibration import load_calibration, project_observation, validate_calibration
from footballer_analyzer.pitch_geometry import pitch_polylines, transform_points
from footballer_analyzer.pitch_models import pitch_landmarks, pitch_spec
from footballer_analyzer.reporting import draw_pitch
from footballer_analyzer.storage import read_json, read_rows, write_json
from footballer_analyzer.video import read_frame


def review(run_dir, reference=None):
    """保存同一固定矩阵在首中末及参考帧的叠加图，独立点仅用于验收。"""
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    calibration = load_calibration(run_dir)
    validate_calibration(calibration, manifest, require_reviewed=False)
    if calibration.get("mode") == "white_lines":
        raise ValueError("此脚本仅复查手动固定标定")
    destination = run_dir / "diagnostics/manual_calibration"
    destination.mkdir(parents=True, exist_ok=True)
    spec = calibration.get("pitch_spec") or {**pitch_spec(), "length_m": calibration["pitch_length_m"], "width_m": calibration["pitch_width_m"]}
    length, width = spec["length_m"], spec["width_m"]
    inverse = np.linalg.inv(calibration["homography"])
    frames = sorted({calibration["start_frame"], (calibration["start_frame"] + calibration["end_frame"] - 1) // 2, calibration["reference_frame"], calibration["end_frame"] - 1})
    tracks = list(read_rows(run_dir / "tracks.parquet"))
    summary = {"pitch_model": spec["model"], "reference_frame": calibration["reference_frame"], "fit_rmse_m": calibration["fit_rmse_m"], "frames": [], "independent_points": []}
    for frame_index in frames:
        image = read_frame(Path(manifest["config"]["video"]), frame_index)
        for _, points in pitch_polylines(spec=spec):
            pixels = transform_points(points, inverse)
            for first, second in zip(pixels[:-1], pixels[1:]):
                if np.isfinite([first, second]).all() and np.abs([first, second]).max() < 1e6:
                    cv2.line(image, tuple(np.rint(first).astype(int)), tuple(np.rint(second).astype(int)), (0, 230, 255), 2, cv2.LINE_AA)
        map_width = 680
        map_height = round(map_width * width / length)
        pitch = np.full((image.shape[0], map_width + 40, 3), 30, np.uint8)
        draw_pitch(pitch, 20, 70, map_width, map_height, length, width, spec)
        rows = [row for row in tracks if row["frame_index"] == frame_index]
        projected = 0
        for row in rows:
            horizontal, vertical, reason = project_observation(row, calibration)
            if reason is not None:
                continue
            projected += 1
            foot = (round((row["bbox_x1"] + row["bbox_x2"]) / 2), round(row["bbox_y2"]))
            cv2.circle(image, foot, 4, (70, 255, 160), -1)
            point = (20 + round(horizontal / length * map_width), 70 + round(vertical / width * map_height))
            cv2.circle(pitch, point, 5, (70, 255, 160), -1)
            for surface, location in [(image, foot), (pitch, point)]:
                cv2.putText(surface, str(row["track_id"]), (location[0] + 5, location[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 255, 160), 1)
        label = f"Frame {frame_index} | static {spec['model']} | {projected}/{len(rows)} on pitch"
        cv2.putText(image, label, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 255), 2)
        path = destination / f"frame-{frame_index}.png"
        if not cv2.imwrite(str(path), np.hstack([image, pitch])):
            raise OSError("无法保存标定对拍图")
        summary["frames"].append({"frame_index": frame_index, "observations": len(rows), "projected": projected, "image": str(path.resolve())})
    if reference is not None:
        payload = read_json(Path(reference))
        if payload["video_sha256"] != manifest["video_sha256"]:
            raise ValueError("独立验收点与源视频不匹配")
        landmarks = {point["id"]: point["pitch"] for point in pitch_landmarks(spec)}
        used = {point["landmark_id"] for point in calibration.get("point_pairs") or []}
        for point in payload["points"]:
            identifier = point["landmark_id"]
            if identifier not in landmarks or type(point.get("frame_index")) is not int or not calibration["start_frame"] <= point["frame_index"] < calibration["end_frame"]:
                raise ValueError("独立验收点的标识或源帧超出标定范围")
            if identifier in used:
                raise ValueError("独立验收点不能复用参与拟合的参考点")
            expected = transform_points([landmarks[identifier]], inverse)[0]
            summary["independent_points"].append({**point, "error_px": float(np.linalg.norm(expected - point["image"]))})
    write_json(destination / "review.json", summary)
    return summary


def main():
    """读取运行目录及可选独立参考点文件，打印诊断产物目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    arguments = parser.parse_args()
    result = review(arguments.run, arguments.reference)
    print(result)


if __name__ == "__main__":
    main()
