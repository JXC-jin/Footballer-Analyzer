"""用人工抽样框复查漏检、关联丢失与编号变化，导出本地标框图。

用法：python scripts/review_tracking.py --run outputs/run_... --reference configs/tracking_review_demo.json。
人工参考框是近似标注，仅统计指定球员的抽样召回，不计算全片 MOTA、IDF1 或精确率。
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from supervision.detection.utils.iou_and_nms import box_iou_batch

from footballer_analyzer.storage import file_hash, read_json, read_rows, write_json
from footballer_analyzer.video import read_frame


def box_of(row):
    """返回一条观测的原图像素 xyxy 框。"""
    return [row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]]


def match_boxes(reference, candidates, minimum_iou=0.3):
    """对人工框与候选框做一对一 IoU 匹配，返回复查代号到候选行的映射。"""
    if not reference or not candidates:
        return {}
    labels = list(reference)
    overlaps = box_iou_batch(np.asarray(list(reference.values()), dtype=float), np.asarray([box_of(row) for row in candidates], dtype=float))
    costs = np.where(overlaps >= minimum_iou, 1 - overlaps, 1e6)
    reference_indices, candidate_indices = linear_sum_assignment(costs)
    return {labels[reference_index]: candidates[candidate_index] for reference_index, candidate_index in zip(reference_indices, candidate_indices) if overlaps[reference_index, candidate_index] >= minimum_iou}


def evaluate_frames(reference_frames, detections, tracks, minimum_iou=0.3):
    """汇总人工抽样帧的漏检和漏关联；检测缓存缺失时明确返回未知。"""
    tracks_by_frame = defaultdict(list)
    detections_by_frame = defaultdict(list)
    for row in tracks:
        tracks_by_frame[row["frame_index"]].append(row)
    for row in detections or []:
        detections_by_frame[row["frame_index"]].append(row)
    per_frame = []
    identities = defaultdict(list)
    for frame_text, reference in sorted(reference_frames.items(), key=lambda item: int(item[0])):
        frame_index = int(frame_text)
        detected = match_boxes(reference, detections_by_frame[frame_index], minimum_iou) if detections is not None else None
        tracked = match_boxes(reference, tracks_by_frame[frame_index], minimum_iou)
        for label, row in tracked.items():
            identities[label].append({"frame_index": frame_index, "track_id": row["track_id"]})
        per_frame.append({
            "frame_index": frame_index, "expected": len(reference),
            "detected": len(detected) if detected is not None else None, "tracked": len(tracked),
            "detection_misses": sorted(set(reference) - set(detected)) if detected is not None else None,
            "association_misses": sorted(set(detected) - set(tracked)) if detected is not None else None,
            "tracking_misses": sorted(set(reference) - set(tracked)),
        })
    return {"minimum_iou": minimum_iou, "expected": sum(row["expected"] for row in per_frame), "detected": sum(row["detected"] for row in per_frame) if detections is not None else None, "tracked": sum(row["tracked"] for row in per_frame), "frames": per_frame, "sampled_id_changes": sum(first["track_id"] != second["track_id"] for rows in identities.values() for first, second in zip(rows, rows[1:])), "identities": dict(identities)}


def render_samples(manifest, reference, detections, tracks, output):
    """输出抽样帧：红框为检测，绿框为轨迹，蓝框为人工参考与复查代号。"""
    output.mkdir(parents=True, exist_ok=True)
    for frame_text, labels in reference["frames"].items():
        frame_index = int(frame_text)
        frame = read_frame(Path(manifest["config"]["video"]), frame_index)
        cv2.putText(frame, "RED detection | GREEN track ID | BLUE manual reference", (30, 40), 0, 0.8, (255, 255, 255), 2)
        for row in detections or []:
            if row["frame_index"] == frame_index:
                left, top, right, bottom = map(int, box_of(row))
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
                cv2.putText(frame, f'{row["confidence"]:.2f}', (left, bottom + 12), 0, 0.4, (0, 0, 255), 1)
        for row in tracks:
            if row["frame_index"] == frame_index:
                left, top, right, bottom = map(int, box_of(row))
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f'#{row["track_id"]}', (left, top - 5), 0, 0.5, (0, 255, 0), 1)
        for label, box in labels.items():
            left, top, right, bottom = map(int, box)
            cv2.rectangle(frame, (left - 2, top - 2), (right + 2, bottom + 2), (255, 130, 30), 1)
            cv2.putText(frame, label, (left, bottom + 24), 0, 0.3, (255, 180, 90), 1)
        if not cv2.imwrite(str(output / f"frame_{frame_index}.jpg"), frame):
            raise OSError("无法写出复查图")


def main():
    """读取完成运行与同源人工参考，写出复查 JSON 和抽样标框图。"""
    parser = argparse.ArgumentParser(description="按人工抽样框复查人物追踪")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--minimum-iou", type=float, default=0.3)
    args = parser.parse_args()
    if not 0 < args.minimum_iou <= 1:
        parser.error("minimum-iou 必须在 (0, 1] 内")
    run_dir = args.run.resolve()
    manifest = read_json(run_dir / "manifest.json")
    reference = read_json(args.reference)
    if manifest["status"] != "complete":
        raise ValueError("只能复查已完成的运行")
    if reference["video_sha256"] != manifest["video_sha256"] or file_hash(Path(manifest["config"]["video"])) != manifest["video_sha256"]:
        raise ValueError("参考标注、运行和源视频摘要不一致")
    for frame_text, labels in reference["frames"].items():
        frame_index = int(frame_text)
        if not manifest["start_frame"] <= frame_index < manifest["end_frame"] or (frame_index - manifest["start_frame"]) % manifest["config"]["stride"]:
            raise ValueError("人工参考帧不在本次运行的采样窗口内")
        for box in labels.values():
            if len(box) != 4 or not np.isfinite(box).all() or box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError("人工参考框必须为有效 xyxy 坐标")
    tracks = list(read_rows(run_dir / "tracks.parquet"))
    detections_path = run_dir / "detections.parquet"
    detections = list(read_rows(detections_path)) if detections_path.exists() else None
    result = evaluate_frames(reference["frames"], detections, tracks, args.minimum_iou)
    result.update(run_id=run_dir.name, reference_sha256=file_hash(args.reference), reference_note=reference.get("note"), video_sha256=manifest["video_sha256"], config=manifest["config"])
    output = run_dir / "diagnostics"
    write_json(output / "review.json", result)
    render_samples(manifest, reference, detections, tracks, output)
    print({key: result[key] for key in ["run_id", "expected", "detected", "tracked", "sampled_id_changes"]})
    print(output)


if __name__ == "__main__":
    main()
