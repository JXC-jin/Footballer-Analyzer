"""导出逐帧白线验收视频、失败区间和抽样图，可用独立人工参考点检验映射。"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from footballer_analyzer.calibration import calibration_directory, load_calibration
from footballer_analyzer.pitch_geometry import pitch_polylines, transform_points
from footballer_analyzer.storage import read_json, write_json
from footballer_analyzer.video import read_frames
from footballer_analyzer.white_line_pipeline import draw_line_overlay


def invalid_intervals(records):
    """将失败帧合并为左闭右开源帧区间。"""
    intervals = []
    for record in records.values():
        if record["status"] == "valid":
            continue
        frame = record["frame_index"]
        if intervals and intervals[-1][1] == frame:
            intervals[-1][1] = frame + 1
        else:
            intervals.append([frame, frame + 1])
    return intervals


def review_run(run_dir, output=None, reference=None, render_video=True):
    """保存全片对拍和首、中、末、最差、失败帧；参考点只用于验收，不参与拟合。"""
    calibration = load_calibration(run_dir)
    if not calibration or calibration.get("mode") != "white_lines":
        raise ValueError("该运行尚无逐帧白线标定，请先执行 calibrate-lines")
    manifest = read_json(run_dir / "manifest.json")
    output = output or calibration_directory(run_dir, calibration) / "review"
    output.mkdir(parents=True, exist_ok=True)
    records = calibration["frames"]
    frames = sorted(int(frame) for frame in records)
    invalid = invalid_intervals(records)
    selected = {frames[0], frames[len(frames) // 2], frames[-1]}
    selected.update(record["frame_index"] for record in sorted(records.values(), key=lambda record: -(record["p90_error_px"] or 0))[:3])
    selected.update(start for start, _ in invalid[:8])
    errors = []
    reference_rows = []
    if reference:
        payload = read_json(reference)
        if payload["video_sha256"] != manifest["video_sha256"]:
            raise ValueError("参考点与源视频摘要不匹配")
        if payload["pitch_length_m"] != calibration["pitch_length_m"] or payload["pitch_width_m"] != calibration["pitch_width_m"]:
            raise ValueError("参考点与标定场地尺寸不一致")
        for entry in payload["frames"]:
            frame_index = entry["frame_index"]
            selected.add(frame_index)
            record = records.get(str(frame_index))
            result = {"frame_index": frame_index, "valid": bool(record and record["homography"]), "point_count": len(entry["image_points"])}
            if result["valid"]:
                projected = transform_points(entry["pitch_points"], np.linalg.inv(record["homography"]))
                pixel_errors = np.linalg.norm(projected - entry["image_points"], axis=1)
                errors.extend(pixel_errors.tolist())
                result.update(errors_px=pixel_errors.tolist(), max_error_px=float(pixel_errors.max()))
            reference_rows.append(result)
    curves = pitch_polylines(calibration["pitch_length_m"], calibration["pitch_width_m"])
    writer = None
    if render_video:
        writer = cv2.VideoWriter(str(output / "white-lines.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), manifest["video"]["fps"], (960, 540))
        if not writer.isOpened():
            raise ValueError("无法创建白线验收视频")
    try:
        for packet in read_frames(Path(manifest["config"]["video"]), calibration["start_frame"], calibration["end_frame"]):
            if writer is None and packet.frame_index not in selected:
                continue
            record = records[str(packet.frame_index)]
            evidence = cv2.imread(str(calibration_directory(run_dir, calibration) / "masks" / f"{packet.frame_index}.png"), cv2.IMREAD_UNCHANGED)
            if evidence is None:
                raise ValueError(f"缺少第 {packet.frame_index} 帧白线证据")
            overlay = draw_line_overlay(packet.image, record, evidence[:, :, 3], curves)
            preview = cv2.resize(overlay, (960, 540))
            if writer is not None:
                writer.write(preview)
            if packet.frame_index in selected:
                cv2.imwrite(str(output / f"frame_{packet.frame_index}.jpg"), preview)
    finally:
        if writer is not None:
            writer.release()
    summary = {**calibration["summary"], "invalid_intervals": invalid, "reference_frames": reference_rows, "reference_median_error_px": float(np.median(errors)) if errors else None, "reference_max_error_px": float(max(errors)) if errors else None}
    write_json(output / "review.json", summary)
    print(f"[review] {output} | {summary['valid_frames']}/{summary['frame_count']} valid | {len(invalid)} invalid intervals", flush=True)
    return summary


def main():
    """解析运行与人工参考点路径，输出可复现的验收产物。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    review_run(args.run.resolve(), args.output, args.reference, not args.no_video)


if __name__ == "__main__":
    main()
