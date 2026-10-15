# -*- coding: utf-8 -*-
"""测量 YOLO CPU/DirectML 推理与光流耗时，保存基准 JSON。

用法：python scripts/bench_cpu.py --frames 20 --imgsz 640,960。
输出 docs/bench/bench_result.json；整场耗时为外推估计，不代表完整流水线实测。
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "bench"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 找一个真实比赛画面当样本（third_party 里的示例视频）
CANDIDATE_VIDEOS = [
    ROOT / "third_party" / "Football-Vision" / "videos" / "Demo_Video.mp4",
    ROOT / "third_party" / "Football-Game-Possession-Tracking" / "input videos" / "goodview.mp4",
    ROOT / "third_party" / "football_analysis" / "input_videos" / "08fd33_4.mp4",
]


def log(msg: str) -> None:
    """打印并立即刷新一条基准进度消息。"""
    print(msg, flush=True)


def load_sample_frame():
    """返回示例视频首帧与来源名称；无可用视频时返回 1080p 合成噪声图。"""
    import cv2
    import numpy as np

    for p in CANDIDATE_VIDEOS:
        if p.exists():
            cap = cv2.VideoCapture(str(p))
            ok, frame = cap.read()
            cap.release()
            if ok:
                log(f"[sample] 使用真实画面 {p.relative_to(ROOT)}  size={frame.shape[1]}x{frame.shape[0]}")
                return frame, str(p.relative_to(ROOT))
    log("[sample] 未找到示例视频，改用合成 1920x1080 噪声图（耗时数字仍可比）")
    return (np.random.rand(1080, 1920, 3) * 255).astype("uint8"), "synthetic"


def timeit(fn, n: int, warmup: int = 3) -> float:
    """先预热 warmup 次，再对 n 次调用计时，返回平均秒数；n 须为正整数。"""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def bench_torch_yolo(frame, sizes, frames_n: int):
    """按输入尺寸测量 YOLO CPU predict，返回含前后处理的毫秒/FPS 记录列表。"""
    import torch
    from ultralytics import YOLO

    results = []
    n_threads = torch.get_num_threads()
    log(f"[torch] {torch.__version__}  threads={n_threads}  cuda={torch.cuda.is_available()}")

    for weights in ["yolov8n.pt", "yolo11n.pt"]:
        try:
            model = YOLO(weights)  # 首次会自动下载权重（约 6MB）
        except Exception as e:  # noqa: BLE001
            log(f"[torch] 跳过 {weights}: {e}")
            continue
        for imgsz in sizes:
            try:
                fn = lambda: model.predict(frame, imgsz=imgsz, device="cpu", verbose=False)  # noqa: E731
                dt = timeit(fn, n=frames_n, warmup=2)
                fps = 1.0 / dt
                log(f"[torch] {weights:12s} imgsz={imgsz:4d}  {dt*1000:8.1f} ms/帧  {fps:6.1f} FPS")
                results.append({"backend": "torch-cpu", "model": weights, "imgsz": imgsz,
                                "ms_per_frame": round(dt * 1000, 1), "fps": round(fps, 1)})
            except Exception as e:  # noqa: BLE001
                log(f"[torch] {weights} imgsz={imgsz} 失败: {e}")
    return results


def bench_onnx(frame, sizes, frames_n: int, weights="yolov8n.pt"):
    """导出 ONNX 并对比可用 CPU/DirectML 后端，返回推理计时记录；失败配置跳过。"""
    import numpy as np
    import onnxruntime as ort
    from ultralytics import YOLO

    results = []
    log(f"[ort] {ort.__version__}  available providers={ort.get_available_providers()}")

    for imgsz in sizes:
        onnx_path = OUT_DIR / f"{Path(weights).stem}_{imgsz}.onnx"
        try:
            if not onnx_path.exists():
                YOLO(weights).export(format="onnx", imgsz=imgsz, opset=12, dynamic=False, simplify=False)
                # ultralytics 会导出到当前工作目录
                produced = Path(weights).with_suffix(".onnx")
                if produced.exists() and produced.resolve() != onnx_path.resolve():
                    produced.replace(onnx_path)
        except Exception as e:  # noqa: BLE001
            log(f"[ort] 导出 {weights} imgsz={imgsz} 失败: {e}")
            continue
        if not onnx_path.exists():
            log(f"[ort] 未找到 {onnx_path.name}，跳过")
            continue

        # 构造 letterbox 后的输入（1,3,S,S），归一化到 0-1
        import cv2
        resized = cv2.resize(frame, (imgsz, imgsz))
        blob = resized[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob)

        for provider in ort.get_available_providers():
            if provider not in ("CPUExecutionProvider", "DmlExecutionProvider"):
                continue
            try:
                so = ort.SessionOptions()
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess = ort.InferenceSession(str(onnx_path), so, providers=[provider])
                name = sess.get_inputs()[0].name
                fn = lambda: sess.run(None, {name: blob})  # noqa: E731
                dt = timeit(fn, n=frames_n, warmup=3)
                tag = "核显(DirectML)" if provider.startswith("Dml") else "CPU"
                log(f"[ort] {tag:14s} imgsz={imgsz:4d}  {dt*1000:8.1f} ms/帧  {1/dt:6.1f} FPS")
                results.append({"backend": f"onnx-{provider}", "model": weights, "imgsz": imgsz,
                                "ms_per_frame": round(dt * 1000, 1), "fps": round(1 / dt, 1)})
            except Exception as e:  # noqa: BLE001
                log(f"[ort] {provider} imgsz={imgsz} 失败: {e}")
    return results


def bench_optical_flow(frame, frames_n: int = 20):
    """对同一图像测量稀疏 LK 与稠密 Farneback 光流，返回毫秒/FPS 记录列表。"""
    import cv2
    import numpy as np

    results = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    mask[int(gray.shape[0] * 0.55):, :] = 255  # 只看下半场（草地），与教学项目一致

    def lk():
        """提取下半图角点并计算稀疏光流；无角点时返回 None。"""
        p0 = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.3, minDistance=3, mask=mask)
        if p0 is None:
            return None
        p1, st, _ = cv2.calcOpticalFlowPyrLK(gray, gray, p0, None)
        return p1

    dt = timeit(lk, n=frames_n, warmup=3)
    log(f"[flow] 稀疏光流(goodFeatures+calcOpticalFlowPyrLK)  {dt*1000:7.1f} ms/帧  {1/dt:6.1f} FPS")
    results.append({"backend": "opencv", "model": "sparse-LK-optical-flow", "imgsz": frame.shape[1],
                    "ms_per_frame": round(dt * 1000, 1), "fps": round(1 / dt, 1)})

    def farneback():
        """计算当前灰度图到自身的稠密光流，供耗时比较。"""
        return cv2.calcOpticalFlowFarneback(gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    try:
        dt2 = timeit(farneback, n=max(5, frames_n // 4), warmup=2)
        log(f"[flow] 稠密光流(Farneback)                        {dt2*1000:7.1f} ms/帧  {1/dt2:6.1f} FPS")
        results.append({"backend": "opencv", "model": "farneback-dense-flow", "imgsz": frame.shape[1],
                        "ms_per_frame": round(dt2 * 1000, 1), "fps": round(1 / dt2, 1)})
    except Exception as e:  # noqa: BLE001
        log(f"[flow] Farneback 失败: {e}")
    return results


def main():
    """解析基准参数，执行可用测试并打印外推耗时，将结果写入 JSON。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=20, help="每个配置测多少帧")
    ap.add_argument("--imgsz", type=str, default="640,960,1280")
    ap.add_argument("--skip-onnx", action="store_true")
    args = ap.parse_args()
    sizes = [int(s) for s in args.imgsz.split(",")]

    log("=" * 78)
    log(f"设备: {platform.processor()}  |  {platform.system()} {platform.release()}")
    log(f"Python: {platform.python_version()}  |  每配置 {args.frames} 帧  |  imgsz={sizes}")
    log("=" * 78)

    frame, sample_name = load_sample_frame()
    all_res = []

    try:
        all_res += bench_torch_yolo(frame, sizes, args.frames)
    except Exception as e:  # noqa: BLE001
        log(f"[torch] 整段失败: {e}")

    if not args.skip_onnx:
        try:
            all_res += bench_onnx(frame, [s for s in sizes if s <= 640] or [640], args.frames)
        except Exception as e:  # noqa: BLE001
            log(f"[ort] 整段失败: {e}")

    try:
        all_res += bench_optical_flow(frame, args.frames)
    except Exception as e:  # noqa: BLE001
        log(f"[flow] 整段失败: {e}")

    # ---- 整场耗时外推 ----
    MATCH_FRAMES = 90 * 60 * 30  # 90 分钟 30fps
    log("\n" + "=" * 78)
    log(f"整场外推（{MATCH_FRAMES} 帧 = 90min @30fps，只算检测一项）")
    log("=" * 78)
    for r in all_res:
        if "imgsz" in r and r["imgsz"] == 640:
            hours = MATCH_FRAMES * r["ms_per_frame"] / 1000 / 3600
            log(f"  {r['backend']:26s} {r['model']:22s} -> {hours:6.1f} 小时/场")
            r["match_hours_est"] = round(hours, 1)

    payload = {"sample": sample_name,
               "machine": {"cpu": platform.processor(), "os": platform.platform()},
               "results": all_res}
    (OUT_DIR / "bench_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n[out] {OUT_DIR / 'bench_result.json'}")


if __name__ == "__main__":
    main()
