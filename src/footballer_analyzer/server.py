"""提供仅监听回环地址的复盘服务：源帧、轨迹、标定、身份编辑与报告生成。"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
import json
import mimetypes
import re
import threading

import cv2

from .calibration import load_calibration, save_calibration
from .config import ROOT
from .identity import load_players, validate_bindings
from .storage import read_json, read_rows, read_yaml, write_yaml
from .video import read_frame

WEB = Path(__file__).parent / "web"


class ReviewApp:
    """管理输出根目录、轨迹缓存和编辑锁，供本地 HTTP 处理器共享。"""

    def __init__(self, output_root):
        """解析输出根目录，初始化可重入锁与轨迹缓存。"""
        self.output_root = Path(output_root).resolve()
        self.lock = threading.RLock()
        self.track_cache = {}

    def run_dir(self, run_id):
        """返回根目录内有效运行的绝对路径；非法 ID、越界或缺清单时抛出 ValueError。"""
        if not re.fullmatch(r"run_[A-Za-z0-9_]+", run_id):
            raise ValueError("运行标识无效")
        candidate = (self.output_root / run_id).resolve()
        if candidate.parent != self.output_root or not (candidate / "manifest.json").is_file():
            raise ValueError("找不到该运行")
        return candidate

    def runs(self):
        """按运行名称倒序返回片段标识、运行状态与处理帧数列表。"""
        entries = []
        for path in sorted(self.output_root.glob("run_*/manifest.json"), reverse=True):
            manifest = read_json(path)
            entries.append({"run_id": manifest["run_id"], "status": manifest["status"], "clip_id": manifest["config"]["clip_id"], "processed_frames": manifest.get("processed_frames", 0)})
        return entries

    def data(self, run_id):
        """返回已完成运行的轨迹、编辑输入和最新完成报告，并标记报告是否过期。"""
        run_dir = self.run_dir(run_id)
        manifest = read_json(run_dir / "manifest.json")
        if manifest["status"] != "complete":
            raise ValueError("该运行未完成，请查看运行清单中的错误")
        tracks_path = run_dir / "tracks.parquet"
        cache_key = (run_id, tracks_path.stat().st_mtime_ns)
        if cache_key not in self.track_cache:
            self.track_cache = {cache_key: list(read_rows(tracks_path))}
        players = load_players(Path(manifest["config"]["players"]))
        bindings = read_yaml(run_dir / "bindings.yaml", {"run_id": run_id, "bindings": []})
        calibration = load_calibration(run_dir)
        reports = []
        report_paths = (run_dir / "reports").glob("report_*/manifest.json")
        for path in sorted(report_paths, key=lambda candidate: (candidate.stat().st_mtime_ns, candidate.parent.name), reverse=True):
            report_manifest = read_json(path)
            if report_manifest["status"] == "complete":
                reports.append(path.parent.name)
        latest = None
        if reports:
            report_dir = run_dir / "reports" / reports[0]
            report_manifest = read_json(report_dir / "manifest.json")
            latest = {
                "report_id": reports[0], "summary": read_json(report_dir / "summary.json"),
                "stale": report_manifest["bindings"] != bindings or report_manifest["calibration"] != calibration or report_manifest["players"] != list(players.values()),
                "motion": list(read_rows(report_dir / "motion.parquet")),
            }
        return {"manifest": manifest, "tracks": self.track_cache[cache_key], "catalog": read_json(run_dir / "catalog.json"), "players": list(players.values()), "bindings": bindings, "calibration": calibration, "reports": reports, "latest_report": latest}

    def save_bindings(self, run_id, payload):
        """校验整份身份区间后原子保存，返回保存状态；冲突或越界时抛错。"""
        run_dir = self.run_dir(run_id)
        manifest = read_json(run_dir / "manifest.json")
        validate_bindings(payload, manifest, load_players(Path(manifest["config"]["players"])), read_json(run_dir / "catalog.json"))
        write_yaml(run_dir / "bindings.yaml", payload)
        return {"saved": True}

    def save_player(self, run_id, payload):
        """校验球员 ID 和显示名，新增或更新共享档案，返回保存状态。"""
        run_dir = self.run_dir(run_id)
        manifest = read_json(run_dir / "manifest.json")
        path = Path(manifest["config"]["players"])
        players = load_players(path)
        player_id = str(payload.get("player_id", "")).strip()
        display_name = str(payload.get("display_name", "")).strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", player_id) or not display_name or len(display_name) > 80:
            raise ValueError("球员 ID 使用 1–64 位字母、数字、下划线或短横线，姓名不得为空")
        players[player_id] = {"player_id": player_id, "display_name": display_name}
        write_yaml(path, {"players": list(players.values())})
        return {"saved": True}


def handler_for(app):
    """创建绑定 ReviewApp 的 HTTP 处理器类，限制可访问路径和写入来源。"""
    class ReviewHandler(BaseHTTPRequestHandler):
        """处理静态页面、本地 JSON 接口、源帧与报告文件请求。"""

        def log_message(self, format_string, *args):
            """仅输出非 200/304 状态的访问日志。"""
            if args and str(args[1]) not in ["200", "304"]:
                super().log_message(format_string, *args)

        def send_bytes(self, payload, content_type, status=200):
            """发送字节响应，显式标记类型和长度，并禁用缓存与内容类型嗅探。"""
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def send_json(self, payload, status=200):
            """发送 UTF-8 JSON 响应，拒绝不可序列化数值。"""
            self.send_bytes(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def check_local(self, mutate=False):
            """校验回环 Host；写请求存在 Origin 时要求同源，否则抛出 ValueError。"""
            host = self.headers.get("Host", "")
            if host not in [f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"]:
                raise ValueError("仅允许本地访问")
            origin = self.headers.get("Origin")
            if mutate and origin and origin != f"http://{host}":
                raise ValueError("拒绝跨来源写入")

        def do_GET(self):
            """读取白名单页面、运行数据、源帧或报告；预期错误返回 JSON，断连忽略。"""
            try:
                self.check_local()
                path = unquote(urlsplit(self.path).path)
                if path == "/api/runs":
                    return self.send_json(app.runs())
                match = re.fullmatch(r"/api/runs/([^/]+)(?:/(.*))?", path)
                if match:
                    run_id, action = match.groups()
                    with app.lock:
                        run_dir = app.run_dir(run_id)
                        if action is None:
                            return self.send_json(app.data(run_id))
                        if action.startswith("frames/"):
                            frame_text = action.removeprefix("frames/")
                            if not frame_text.isdigit():
                                raise ValueError("帧号必须为整数")
                            frame_index = int(frame_text)
                            manifest = read_json(run_dir / "manifest.json")
                            if not manifest["start_frame"] <= frame_index < manifest["end_frame"]:
                                raise ValueError("帧号超出分析区间")
                            frame = read_frame(Path(manifest["config"]["video"]), frame_index)
                            success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            if not success:
                                raise ValueError("帧编码失败")
                            return self.send_bytes(encoded.tobytes(), "image/jpeg")
                        if action.startswith("crops/"):
                            file_name = action.removeprefix("crops/")
                            if not re.fullmatch(r"\d+\.jpg", file_name):
                                raise ValueError("缩略图名称无效")
                            return self.send_bytes((run_dir / "crops" / file_name).read_bytes(), "image/jpeg")
                report_match = re.fullmatch(r"/reports/([^/]+)/([^/]+)/([^/]+)", path)
                if report_match:
                    run_id, report_id, file_name = report_match.groups()
                    if not re.fullmatch(r"report_[A-Za-z0-9_]+", report_id) or not re.fullmatch(r"(?:report\.html|[a-f0-9]+\.png|summary\.parquet|review\.mp4)", file_name):
                        raise ValueError("报告路径无效")
                    report_path = app.run_dir(run_id) / "reports" / report_id / file_name
                    return self.send_bytes(report_path.read_bytes(), mimetypes.guess_type(file_name)[0] or "application/octet-stream")
                static_paths = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
                if path in static_paths:
                    static_path = WEB / static_paths[path]
                    return self.send_bytes(static_path.read_bytes(), (mimetypes.guess_type(static_path.name)[0] or "text/plain") + "; charset=utf-8")
                self.send_json({"error": "页面不存在"}, 404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (ValueError, KeyError, TypeError, OSError) as error:
                self.send_json({"error": str(error)}, 400)

        def do_POST(self):
            """接收最多 1 MiB 的 JSON 对象，串行保存编辑或生成报告，预期错误返回 400。"""
            try:
                self.check_local(mutate=True)
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("请求必须使用 application/json")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024 * 1024:
                    raise ValueError("请求长度无效")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("请求必须为 JSON 对象")
                match = re.fullmatch(r"/api/runs/([^/]+)/(calibration|bindings|players|report)", urlsplit(self.path).path)
                if not match:
                    return self.send_json({"error": "接口不存在"}, 404)
                run_id, action = match.groups()
                with app.lock:
                    if action == "calibration":
                        result = save_calibration(app.run_dir(run_id), payload)
                    elif action == "bindings":
                        result = app.save_bindings(run_id, payload)
                    elif action == "players":
                        result = app.save_player(run_id, payload)
                    else:
                        from .reporting import create_report

                        result = {"report_id": create_report(app.run_dir(run_id)).name}
                self.send_json(result)
            except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
                self.send_json({"error": str(error)}, 400)

    return ReviewHandler


def serve(output_root=ROOT / "outputs", port=8765):
    """在 127.0.0.1 的指定端口阻塞运行复盘服务，收到 Ctrl+C 后关闭。"""
    if not 1 <= port <= 65535:
        raise ValueError("端口无效")
    app = ReviewApp(output_root)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_for(app))
    server.daemon_threads = True
    print(f"[review] http://127.0.0.1:{port}  |  Ctrl+C 停止", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
