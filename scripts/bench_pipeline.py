# -*- coding: utf-8 -*-
"""测量示例视频解码、CPU 检测和 ByteTrack 的端到端吞吐。

用法：python scripts/bench_pipeline.py --frames 300。
输出 docs/bench/pipeline_result.json；整场耗时外推按源视频 30 FPS 假设计算。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "bench"
OUT.mkdir(parents=True, exist_ok=True)
VIDEO = ROOT / "third_party" / "Football-Vision" / "videos" / "Demo_Video.mp4"


def main():
    """处理指定数量的采样帧，汇总检测框、轨迹数、吞吐与估计耗时并写入 JSON。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--stride", type=int, default=1, help="每 N 帧处理一帧（vid_stride）")
    args = ap.parse_args()

    from ultralytics import YOLO

    assert VIDEO.exists(), f"缺少样本视频 {VIDEO}"
    model = YOLO(args.model)
    n = args.frames

    print(f"[run] 模型={args.model} imgsz={args.imgsz} vid_stride={args.stride} 帧数={n}", flush=True)
    t0 = time.perf_counter()
    results = model.track(
        source=str(VIDEO), stream=True, persist=True, tracker="bytetrack.yaml",
        device="cpu", imgsz=args.imgsz, vid_stride=args.stride, verbose=False,
    )
    seen, n_boxes, n_ids = 0, 0, set()
    for r in results:
        seen += 1
        if r.boxes is not None:
            n_boxes += len(r.boxes)
            if r.boxes.id is not None:
                n_ids.update(int(i) for i in r.boxes.id.tolist())
        if seen >= n:
            break
    dt = time.perf_counter() - t0

    processed_video_seconds = seen * args.stride / 30.0   # 假设源 30fps
    realtime_factor = processed_video_seconds / dt        # >1 表示比实时快
    match_hours = (90 * 60) / realtime_factor / 3600      # 整场 90 分钟视频的预计处理时长
    # 平均每帧球员数（说明画面里到底有多少目标）
    per_frame = n_boxes / max(seen, 1)

    print(f"[run] 处理 {seen} 帧（源视频 {processed_video_seconds/60:.2f} 分钟）耗时 {dt:.1f}s")
    print(f"[run] 吞吐 {seen/dt:.1f} FPS | 相对实时倍率 {realtime_factor:.2f}x")
    print(f"[run] 平均每帧检出框 {per_frame:.1f} 个，累计出现 track id {len(n_ids)} 个")
    print(f"[run] >>> 整场 90 分钟视频预计处理时长: {match_hours:.1f} 小时")

    payload = {
        "config": vars(args),
        "frames_processed": seen,
        "wall_seconds": round(dt, 1),
        "throughput_fps": round(seen / dt, 1),
        "realtime_factor": round(realtime_factor, 2),
        "match_90min_hours": round(match_hours, 1),
        "avg_boxes_per_frame": round(per_frame, 1),
        "unique_track_ids": len(n_ids),
    }
    (OUT / "pipeline_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {OUT / 'pipeline_result.json'}")


if __name__ == "__main__":
    main()
