"""复用原始轨迹生成版本化跑动报告，并对显式选择的历史报告执行防重复汇总。"""

from collections import Counter, defaultdict
from pathlib import Path
import html
import time

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .calibration import load_calibration, project_observation, validate_calibration
from .identity import load_players, resolve_identity, validate_bindings
from .metrics import MOTION_SCHEMA, SUMMARY_SCHEMA, calculate_motion, summarize
from .pitch_geometry import pitch_polylines
from .storage import RowWriter, file_hash, new_id, read_json, read_rows, read_yaml, write_json, write_yaml
from .video import read_frames


def draw_pitch(image, left, top, width, height, length_m, width_m, spec=None, model="eleven"):
    """在像素区域按保存的场地几何绘图，与前端使用相同米制折线。"""
    cv2.rectangle(image, (left, top), (left + width, top + height), (38, 75, 49), -1)
    for _, points in pitch_polylines(length_m, width_m, model=model, spec=spec):
        pixels = np.rint(points * [width / length_m, height / width_m] + [left, top]).astype(np.int32)
        pixels[:, 0] = np.clip(pixels[:, 0], left, left + width)
        pixels[:, 1] = np.clip(pixels[:, 1], top, top + height)
        cv2.polylines(image, [pixels], False, (170, 200, 180), 1, cv2.LINE_AA)


def render_review(run_dir, report_dir, manifest, calibration, bindings):
    """顺序解码源视频，将轨迹 ID、身份、脚点和球场位置写入 review.mp4。"""
    info = manifest["video"]
    rows_by_frame = defaultdict(list)
    for row in read_rows(run_dir / "tracks.parquet"):
        rows_by_frame[row["frame_index"]].append(row)
    writer = cv2.VideoWriter(str(report_dir / "review.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"] / manifest["config"]["stride"], (info["width"], info["height"]))
    if not writer.isOpened():
        raise ValueError("无法创建报告视频")
    pitch_width = min(420, info["width"] // 3)
    pitch_height = round(pitch_width * calibration["pitch_width_m"] / calibration["pitch_length_m"])
    try:
        for packet in read_frames(Path(manifest["config"]["video"]), manifest["start_frame"], manifest["end_frame"], manifest["config"]["stride"]):
            frame = packet.image
            draw_pitch(frame, 20, 20, pitch_width, pitch_height, calibration["pitch_length_m"], calibration["pitch_width_m"], calibration.get("pitch_spec"), calibration.get("pitch_model", "eleven"))
            for row in rows_by_frame[packet.frame_index]:
                player_id, team_id, _, reason = resolve_identity(row, bindings)
                if reason == "excluded":
                    continue
                color = (80, 210, 255) if team_id == "home" else (255, 140, 95) if team_id == "away" else (170, 180, 185)
                left, top, right, bottom = [int(row[key]) for key in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]]
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                foot = ((left + right) // 2, bottom)
                cv2.circle(frame, foot, 4, color, -1)
                cv2.putText(frame, f'#{row["track_id"]} {player_id or "unbound"}', (left, max(15, top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                world_x, world_y, projection_reason = project_observation(row, calibration)
                if world_x is not None and 0 <= world_x <= calibration["pitch_length_m"] and 0 <= world_y <= calibration["pitch_width_m"]:
                    point = (20 + round(world_x / calibration["pitch_length_m"] * pitch_width), 20 + round(world_y / calibration["pitch_width_m"] * pitch_height))
                    cv2.circle(frame, point, 4, color, -1 if projection_reason is None else 1)
                    cv2.putText(frame, str(row["track_id"]), (point[0] + 4, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            writer.write(frame)
    finally:
        writer.release()


def render_charts(report_dir, motions, summary, players, calibration):
    """为每名汇总球员保存速度曲线与按有效秒数加权的热图；断段分别绘制。"""
    for summary_row in summary:
        player_id = summary_row["player_id"]
        player_rows = [row for row in motions if row["player_id"] == player_id]
        figure, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
        segments = defaultdict(list)
        for row in player_rows:
            if row["valid"]:
                segments[row["segment_id"]].append(row)
        for segment_rows in segments.values():
            speeds = [row for row in segment_rows if row["speed_mps"] is not None]
            axes[0].plot([row["timestamp_s"] for row in speeds], [row["speed_mps"] * 3.6 for row in speeds], color="#118a79", linewidth=1.5)
        axes[0].set(xlabel="Source time (s)", ylabel="Speed (km/h)", title="Observed speed / gaps kept")
        heat_rows = [row for row in player_rows if row["delta_t_s"] is not None and row["valid"]]
        heatmap, _, _ = np.histogram2d(
            [row["x_smooth_m"] for row in heat_rows], [row["y_smooth_m"] for row in heat_rows],
            bins=(35, 23), range=[[0, calibration["pitch_length_m"]], [0, calibration["pitch_width_m"]]],
            weights=[row["delta_t_s"] for row in heat_rows],
        )
        axes[1].imshow(heatmap.T, extent=[0, calibration["pitch_length_m"], calibration["pitch_width_m"], 0], cmap="YlGn", aspect="equal")
        for _, points in pitch_polylines(calibration["pitch_length_m"], calibration["pitch_width_m"], spec=calibration.get("pitch_spec"), model=calibration.get("pitch_model", "eleven")):
            axes[1].plot(points[:, 0], points[:, 1], color="#668466", linewidth=0.6)
        axes[1].set(xlabel="Pitch x (m)", ylabel="Pitch y (m)", title="Time-weighted occupancy (s)")
        axes[1].set_xlim(0, calibration["pitch_length_m"])
        axes[1].set_ylim(calibration["pitch_width_m"], 0)
        figure.suptitle(player_id)
        figure.savefig(report_dir / f"{player_id.encode('utf-8').hex()}.png", dpi=120)
        plt.close(figure)


def create_report(run_dir: Path):
    """校验源视频、已确认标定和身份绑定，生成新报告目录并返回路径。

    保留原始轨迹与旧报告，保存输入快照、统计、图表和视频；生成失败记录状态并抛错。
    """
    manifest = read_json(run_dir / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("跟踪尚未成功完成")
    if file_hash(Path(manifest["config"]["video"])) != manifest["video_sha256"]:
        raise ValueError("源视频发生变化，请重新跟踪")
    calibration = load_calibration(run_dir)
    validate_calibration(calibration, manifest)
    players_path = Path(manifest["config"]["players"])
    players = load_players(players_path)
    binding_payload = read_yaml(run_dir / "bindings.yaml", {"run_id": manifest["run_id"], "bindings": []})
    bindings = validate_bindings(binding_payload, manifest, players, read_json(run_dir / "catalog.json"))
    if not any(entry["action"] == "bind" for entry in bindings):
        raise ValueError("请至少绑定一名球员后再生成个人报告")
    report_dir = run_dir / "reports" / new_id("report")
    report_dir.mkdir(parents=True)
    started = time.perf_counter()
    report_manifest = {
        "schema_version": 1, "report_id": report_dir.name, "run_id": manifest["run_id"], "status": "running",
        "source": manifest, "tracks_sha256": file_hash(run_dir / "tracks.parquet"),
        "calibration": calibration, "bindings": binding_payload, "players": list(players.values()),
    }
    write_json(report_dir / "manifest.json", report_manifest)
    write_yaml(report_dir / "calibration.yaml", calibration)
    write_yaml(report_dir / "bindings.yaml", binding_payload)
    write_yaml(report_dir / "players.yaml", {"players": list(players.values())})
    try:
        motions = []
        with RowWriter(report_dir / "motion.parquet", MOTION_SCHEMA) as writer:
            for row in calculate_motion(read_rows(run_dir / "tracks.parquet"), manifest, calibration, bindings):
                writer.append(row)
                motions.append(row)
        summary = summarize(motions, manifest, bindings)
        with RowWriter(report_dir / "summary.parquet", SUMMARY_SCHEMA) as writer:
            for row in summary:
                writer.append(row)
        quality = dict(Counter(row["reason"] or "valid" for row in motions))
        write_json(report_dir / "summary.json", {"players": summary, "quality": quality})
        render_charts(report_dir, motions, summary, players, calibration)
        analysis_s = time.perf_counter() - started
        render_started = time.perf_counter()
        render_review(run_dir, report_dir, manifest, calibration, bindings)
        report_manifest.update(status="complete", analysis_s=round(analysis_s, 3), render_s=round(time.perf_counter() - render_started, 3), quality=quality)
        write_json(report_dir / "manifest.json", report_manifest)
        sections = []
        for row in summary:
            name = html.escape(players[row["player_id"]]["display_name"])
            distance = f'{row["distance_m"]:.2f} m' if row["distance_m"] is not None else "数据不足"
            speed = f'{row["avg_speed_mps"] * 3.6:.2f} km/h' if row["avg_speed_mps"] is not None else "数据不足"
            sections.append(f'<section><h2>{name} · {html.escape(row["player_id"])}</h2><p>已观测距离 {distance} · 平均速度 {speed} · 有效 {row["valid_seconds"]:.2f} s · 窗口覆盖 {row["coverage"]:.1%}</p><img alt="速度曲线与时间加权热图" src="{row["player_id"].encode("utf-8").hex()}.png"></section>')
        report_html = f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>球员跑动报告</title><style>body{{font:16px system-ui;max-width:1120px;margin:40px auto;padding:0 24px;color:#183a35;background:#f3f6f3}}section{{background:white;padding:24px;margin:20px 0;border-radius:12px}}img{{max-width:100%}}code{{overflow-wrap:anywhere}}</style><h1>球员跑动报告</h1><p>{html.escape(manifest["config"]["match_id"])} / {html.escape(manifest["config"]["clip_id"])}</p><p>仅统计可靠区域内的有效观测，缺口不插值、不外推整场。±10% 精度尚未验证。</p><p><a href="review.mp4">下载叠加视频</a> · <a href="summary.parquet">统计数据</a></p>{"".join(sections)}<h2>质量统计</h2><code>{html.escape(str(quality))}</code></html>'
        (report_dir / "report.html").write_text(report_html, encoding="utf-8")
        print(f"[report] {report_dir}", flush=True)
        return report_dir
    except Exception as error:
        report_manifest.update(status="failed", error=str(error))
        write_json(report_dir / "manifest.json", report_manifest)
        raise


def history(player_id, report_dirs):
    """查询指定球员在选定报告中的记录，返回片段列表与可累计的米制距离。

    拒绝重复版本和已知重叠片段；同场偏移未知时仅逐片展示，累计距离返回 None。
    """
    selected = []
    seen_clips = set()
    for report_dir in report_dirs:
        report_dir = Path(report_dir)
        report_manifest = read_json(report_dir / "manifest.json")
        if report_manifest["status"] != "complete":
            raise ValueError("历史查询包含未完成报告")
        source = report_manifest["source"]
        clip_key = (source["config"]["match_id"], source["config"]["clip_id"])
        if clip_key in seen_clips:
            raise ValueError("同一片段选择了多个报告版本，拒绝重复累计")
        seen_clips.add(clip_key)
        for prior_source, _ in selected:
            same_video = source["video_sha256"] == prior_source["video_sha256"]
            if same_video and max(source["start_frame"], prior_source["start_frame"]) < min(source["end_frame"], prior_source["end_frame"]):
                raise ValueError("报告使用了重叠的源视频帧，拒绝重复累计")
            same_period = (source["config"]["match_id"], source["config"]["period"]) == (prior_source["config"]["match_id"], prior_source["config"]["period"])
            offset = source["config"]["match_offset_s"]
            prior_offset = prior_source["config"]["match_offset_s"]
            if same_period and offset is not None and prior_offset is not None:
                begin = offset + source["start_frame"] / source["video"]["fps"]
                end = offset + source["end_frame"] / source["video"]["fps"]
                prior_begin = prior_offset + prior_source["start_frame"] / prior_source["video"]["fps"]
                prior_end = prior_offset + prior_source["end_frame"] / prior_source["video"]["fps"]
                if max(begin, prior_begin) < min(end, prior_end):
                    raise ValueError("同场同半场片段重叠，拒绝累计")
        selected.append((source, list(read_rows(report_dir / "summary.parquet"))))
    results = [row for _, rows in selected for row in rows if row["player_id"] == player_id]
    match_counts = Counter(source["config"]["match_id"] for source, _ in selected)
    cumulative_allowed = all(match_counts[source["config"]["match_id"]] <= 1 or source["config"]["match_offset_s"] is not None for source, _ in selected)
    return {"player_id": player_id, "records": results, "label": "已分析片段累计", "distance_m": sum(row["distance_m"] or 0 for row in results) if cumulative_allowed else None, "note": "同场片段时间偏移未知，仅逐片展示" if not cumulative_allowed else "仅累计有效、无重复的已分析片段"}
