# -*- coding: utf-8 -*-
"""
素材体检：判断一段视频能否用于本项目的分析流水线。
================================================================
回答三个问题：
  1. 镜头怎么动？（静止 / 小幅随球平移 / 大幅摇摄 / 有变焦旋转）
  2. 球员有多大？（决定检测与身份是否可行）
  3. 镜头运动给"像素→米制"换算注入多少误差？（无需标定的估算）

核心判据（不需要相机标定）：
  用球员身高当尺子(约 1.8 m) -> 1px ≈ 1.8/h 米（h = 球员框高中位数）
  相对误差 ≈ (每帧镜头位移px × 1.8/h) / (球员速度 / 帧率)
  <=10% 理想 | <=20% 可接受 | >30% 单一单应性不可用，需逐帧补偿或动态标定

用法:
  .venv\\Scripts\\python.exe scripts\\inspect_clip.py --video data/clips/xxx.mp4
  .venv\\Scripts\\python.exe scripts\\inspect_clip.py --video xxx.mp4 --no-detect   # 跳过检测，只测镜头
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "bench"


def read_video_info(cap):
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return w, h, fps, n


def analyze_camera(video: str, sample_step: int = 1, work_w: int = 960):
    """逐帧估计全局运动，分解为 平移 / 旋转 / 缩放。返回原始分辨率下的像素量。"""
    cap = cv2.VideoCapture(video)
    w, h, fps, n = read_video_info(cap)
    scale = w / work_w

    prev_gray, prev_pts = None, None
    tx_list, ty_list, rot_list, scale_list, resid_list = [], [], [], [], []

    for i in range(0, n, sample_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(fr, (work_w, int(h / scale))), cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            p1, stt, _ = cv2.calcOpticalFlowPyrLK(prev_gray, g, prev_pts, None)
            good = stt.ravel() == 1
            if good.sum() >= 8:
                M, _ = cv2.estimateAffinePartial2D(prev_pts[good], p1[good], method=cv2.RANSAC)
                if M is not None:
                    a, b = M[0, 0], M[0, 1]
                    tx_list.append(M[0, 2] * scale)
                    ty_list.append(M[1, 2] * scale)
                    scale_list.append(float(np.hypot(a, b)))
                    rot_list.append(float(np.degrees(np.arctan2(b, a))))
                    # 平移对齐后的残差（越小越说明"纯平移"）
                    M_t = np.float32([[1, 0, -M[0, 2]], [0, 1, -M[1, 2]]])
                    warped = cv2.warpAffine(prev_gray, M_t, (g.shape[1], g.shape[0]))
                    a1 = warped - warped.mean()
                    a2 = g - g.mean()
                    denom = (np.linalg.norm(a1) * np.linalg.norm(a2))
                    resid_list.append(float(1 - (a1 * a2).sum() / denom) if denom > 1e-6 else 1.0)
        prev_gray = g
        prev_pts = cv2.goodFeaturesToTrack(g, maxCorners=400, qualityLevel=0.01, minDistance=8)
    cap.release()

    if not tx_list:
        return None
    step = sample_step
    per_frame_shift = [float(np.hypot(x, y)) / step for x, y in zip(tx_list, ty_list)]
    cum = float(sum(np.hypot(x, y) for x, y in zip(tx_list, ty_list)))
    zoom = [abs(s - 1.0) for s in scale_list]
    return {
        "resolution": f"{w}x{h}", "fps": fps, "frames": n, "duration_s": round(n / fps, 1),
        "sampled_pairs": len(tx_list),
        "per_frame_shift_px_median": round(st.median(per_frame_shift), 3),
        "per_frame_shift_px_p90": round(float(np.percentile(per_frame_shift, 90)), 3),
        "cumulative_drift_px": round(cum, 1),
        "zoom_per_frame_median": round(st.median(zoom), 5),
        "rotation_deg_per_frame_median": round(abs(st.median(rot_list)), 4),
        "residual_after_translation_median": round(st.median(resid_list), 4),
        "total_rotation_deg": round(sum(abs(r) for r in rot_list) / step * step, 2),
    }


def analyze_players(video: str, sample_step: int = 5, imgsz: int = 640, conf: float = 0.25):
    from ultralytics import YOLO
    wts = ROOT / "models" / "yolov8n.pt"
    model = YOLO(str(wts) if wts.exists() else "yolov8n.pt")
    cap = cv2.VideoCapture(video)
    w, h, fps, n = read_video_info(cap)
    heights, counts, ball = [], [], 0
    for i in range(0, n, sample_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            break
        r = model.predict(fr, imgsz=imgsz, conf=conf, device="cpu", verbose=False)[0]
        c = 0
        for b in r.boxes:
            nm = model.names[int(b.cls)]
            if nm == "person":
                y1, y2 = b.xyxy[0][1].item(), b.xyxy[0][3].item()
                heights.append(y2 - y1); c += 1
            elif nm == "sports ball":
                ball += 1
        counts.append(c)
    cap.release()
    hs = sorted(heights) if heights else []
    return {
        "sampled_frames": len(counts),
        "persons_per_frame_mean": round(st.mean(counts), 1) if counts else 0,
        "persons_per_frame_min": min(counts) if counts else 0,
        "persons_per_frame_max": max(counts) if counts else 0,
        "bbox_h_median_px": round(st.median(hs), 0) if hs else 0,
        "bbox_h_p10_px": round(float(np.percentile(hs, 10)), 0) if hs else 0,
        "bbox_h_min_px": round(hs[0], 0) if hs else 0,
        "frac_h_ge_30px": round(sum(1 for x in hs if x >= 30) / len(hs), 3) if hs else 0,
        "sports_ball_frames": ball,
    }


def verdict(cam, ply, speed_kmh=18.0):
    """speed_kmh: 用来估算"典型跑动"的参考速度（成年业余球员常见区间 12-25）。"""
    if not cam:
        return "无法分析镜头运动"
    h_px = (ply or {}).get("bbox_h_median_px") or 0
    fps = cam["fps"]
    shift = cam["per_frame_shift_px_median"]
    if h_px <= 0:
        return "缺少球员尺寸，无法给出精度估算"
    m_per_px = 1.8 / h_px                      # 用身高当尺子
    induced_m = shift * m_per_px               # 每帧被镜头运动注入的米数
    v_ms = speed_kmh / 3.6
    player_m = v_ms / fps                      # 球员每帧自己跑的米数
    err = induced_m / player_m if player_m > 0 else 9.9
    lines = [
        f"1px ≈ {m_per_px:.4f} m（由球员框高中位 {h_px:.0f}px 反推）",
        f"镜头运动注入 {induced_m:.4f} m/帧 ; 球员({speed_kmh:.0f}km/h)自身位移 {player_m:.4f} m/帧",
        f"==> 相对误差 ≈ {err*100:.1f}%",
    ]
    if err <= 0.10:
        lines.append("判定: 可用 —— 单一单应性即可，甚至可忽略补偿")
    elif err <= 0.20:
        lines.append("判定: 可用（需一次平移补偿）—— 把各帧归算回参考帧后再套单一 H")
    elif err <= 0.30:
        lines.append("判定: 勉强 —— 建议做平移补偿并只统计近端区域")
    else:
        lines.append("判定: 单一单应性不可用 —— 需逐帧补偿/动态标定，或另找素材")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--no-detect", action="store_true")
    ap.add_argument("--sample-step", type=int, default=1, help="镜头分析每隔几帧取一帧")
    ap.add_argument("--detect-step", type=int, default=5, help="检测每隔几帧取一帧")
    ap.add_argument("--speed", type=float, default=18.0, help="参考跑动速度 km/h")
    args = ap.parse_args()

    print("=" * 74)
    print(f"素材体检: {args.video}")
    print("=" * 74)
    cam = analyze_camera(args.video, args.sample_step)
    if cam:
        print("\n[镜头运动]")
        for k, v in cam.items():
            print(f"  {k:34s} {v}")
        kind = ("静止" if cam["per_frame_shift_px_median"] < 0.05 else
                "小幅随动" if cam["per_frame_shift_px_median"] < 1.5 else "大幅摇摄")
        extra = []
        if cam["zoom_per_frame_median"] > 0.002:
            extra.append("存在变焦")
        if cam["rotation_deg_per_frame_median"] > 0.05:
            extra.append("存在旋转/滚转")
        print(f"  -> 类型: {kind}" + ("；" + "、".join(extra) if extra else "；以平移为主"))

    ply = None
    if not args.no_detect:
        print("\n[球员检测] (COCO 预训练 yolov8n)")
        ply = analyze_players(args.video, args.detect_step)
        for k, v in ply.items():
            print(f"  {k:34s} {v}")

    print("\n[精度估算]")
    print("  " + verdict(cam, ply, args.speed).replace("\n", "\n  "))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / (Path(args.video).stem + "_inspect.json")
    out.write_text(json.dumps({"video": args.video, "camera": cam, "players": ply},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
