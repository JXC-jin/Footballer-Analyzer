"""启动临时回环 HTTP 服务，验证源帧读取以及路径、Host 和写入来源边界。"""

from http.server import ThreadingHTTPServer
from pathlib import Path
import json
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from footballer_analyzer.server import ReviewApp, handler_for
from footballer_analyzer.storage import DETECTION_SCHEMA, RowWriter, read_rows, read_yaml, write_json, write_yaml
import test_reporting


class ServerTests(unittest.TestCase):
    """本地复盘 API 的读操作与输入边界测试。"""

    def test_clip_preview_and_analysis_http_routes(self):
        """验证素材预览、后台状态与启动参数路由，拒绝素材越界和跨源启动。"""
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, manifest = test_reporting.ReportingTests().prepare_run(temporary)
            video = Path(manifest["config"]["video"])
            app = ReviewApp(temporary, video.parent)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(app))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base + "/api/clips") as response:
                    self.assertEqual(json.load(response)[0]["clip_id"], video.name)
                with urlopen(base + f"/api/clips/{video.name}/preview") as response:
                    self.assertEqual(response.headers["Content-Type"], "image/jpeg")
                with urlopen(base + "/api/analysis-job") as response:
                    self.assertEqual(json.load(response)["status"], "idle")
                headers = {"Content-Type": "application/json"}
                body = json.dumps({"clip_id": video.name, "start_frame": 2}).encode()
                with patch.object(app, "start_analysis", return_value={"status": "running"}) as start:
                    with urlopen(Request(base + "/api/analysis", data=body, headers=headers)) as response:
                        self.assertEqual(json.load(response)["status"], "running")
                    start.assert_called_once_with({"clip_id": video.name, "start_frame": 2})
                for request in [Request(base + "/api/clips/..%5Csource.avi/preview"), Request(base + "/api/analysis", data=body, headers={**headers, "Origin": "https://example.com"})]:
                    with self.assertRaises(HTTPError):
                        urlopen(request)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_detection_cache_is_optional_and_refreshes(self):
        """旧运行返回未知检测缓存，缓存补齐后返回原始框且保留置信度。"""
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, manifest = test_reporting.ReportingTests().prepare_run(temporary)
            app = ReviewApp(temporary)
            self.assertIsNone(app.data(run_dir.name)["detections"])
            raw_row = next(read_rows(run_dir / "tracks.parquet"))
            raw_row["confidence"] = 0.15
            with RowWriter(run_dir / "detections.parquet", DETECTION_SCHEMA) as writer:
                writer.append(raw_row)
            refreshed = app.data(run_dir.name)
            self.assertEqual(refreshed["detections"][0]["confidence"], 0.15)
            self.assertEqual(len(refreshed["tracks"]), 10)

    def test_ball_cache_refreshes_without_changing_people(self):
        """区分旧运行与空球缓存，新增、替换、删除球缓存均使响应更新。"""
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, _ = test_reporting.ReportingTests().prepare_run(temporary)
            app = ReviewApp(temporary)
            original = app.data(run_dir.name)
            self.assertIsNone(original["ball_detections"])
            ball_path = run_dir / "ball_detections.parquet"
            with RowWriter(ball_path, DETECTION_SCHEMA):
                pass
            self.assertEqual(app.data(run_dir.name)["ball_detections"], [])
            row = dict(original["tracks"][0], class_id=32, confidence=0.65)
            with RowWriter(ball_path, DETECTION_SCHEMA) as writer:
                writer.append(row)
            refreshed = app.data(run_dir.name)
            self.assertEqual(refreshed["ball_detections"][0]["class_id"], 32)
            self.assertEqual(refreshed["ball_detections"][0]["confidence"], 0.65)
            with RowWriter(ball_path, DETECTION_SCHEMA) as writer:
                writer.append({**row, "confidence": 0.8})
            replaced = app.data(run_dir.name)
            self.assertEqual(replaced["ball_detections"][0]["confidence"], 0.8)
            self.assertEqual(replaced["tracks"], original["tracks"])
            self.assertEqual(refreshed["detections"], original["detections"])
            # 仅移除本测试临时目录中刚创建的缓存，验证旧运行语义恢复。
            self.assertEqual(ball_path.resolve().parent, run_dir.resolve())
            ball_path.unlink()
            self.assertIsNone(app.data(run_dir.name)["ball_detections"])

    def test_local_api_frame_and_input_boundaries(self):
        """验证运行列表与 JPEG 帧可读，越界帧、非法路径、跨源写入和错误 Host 被拒绝。"""
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, manifest = test_reporting.ReportingTests().prepare_run(temporary)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(ReviewApp(temporary)))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base + "/api/runs") as response:
                    self.assertEqual(json.load(response)[0]["run_id"], "run_test")
                with urlopen(base + "/api/runs/run_test") as response:
                    self.assertIsNone(json.load(response)["ball_detections"])
                with urlopen(base + "/api/runs/run_test/frames/4") as response:
                    frame = cv2.imdecode(np.frombuffer(response.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
                    self.assertEqual(frame.shape[:2], (48, 64))
                for suffix in ["../manifest.json", "run_test/frames/99"]:
                    with self.assertRaises(HTTPError):
                        urlopen(base + "/api/runs/" + suffix)
                foreign_request = Request(base + "/api/runs/run_test/players", data=b'{}', headers={"Content-Type": "application/json", "Origin": "https://example.com"})
                with self.assertRaises(HTTPError) as raised:
                    urlopen(foreign_request)
                self.assertIn("跨来源", raised.exception.read().decode())
                bad_host_request = Request(base + "/api/runs", headers={"Host": "example.com"})
                with self.assertRaises(HTTPError):
                    urlopen(bad_host_request)
            finally:
                server.shutdown()
                server.server_close()

    def test_white_line_evidence_and_review_are_bound_to_calibration_version(self):
        """透明掩膜绑定标定版本，旧版本不可误确认，非法版本路径被拒绝。"""
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, _ = test_reporting.ReportingTests().prepare_run(temporary)
            calibration = read_yaml(run_dir / "calibration.yaml")
            calibration.update(mode="white_lines", calibration_id="lines_test", reviewed=False)
            version_dir = run_dir / "calibrations/lines_test"
            write_json(version_dir / "frames.json", {"0": {"status": "insufficient_geometry", "homography": None}})
            write_yaml(version_dir / "calibration.yaml", calibration)
            write_yaml(run_dir / "calibration.yaml", calibration)
            (version_dir / "masks").mkdir()
            cv2.imwrite(str(version_dir / "masks/0.png"), np.zeros((48, 64, 4), np.uint8))
            app = ReviewApp(temporary)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(app))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}/api/runs/run_test"
            try:
                with urlopen(base + "/white-lines/0.png?version=lines_test") as response:
                    self.assertEqual(response.headers["Content-Type"], "image/png")
                with self.assertRaises(HTTPError):
                    urlopen(base + "/white-lines/0.png?version=../../manifest")
                with self.assertRaisesRegex(ValueError, "版本"):
                    app.review_calibration("run_test", {"calibration_id": "lines_old", "reviewed": True})
                app.review_calibration("run_test", {"calibration_id": "lines_test", "reviewed": True})
                self.assertTrue(app.data("run_test")["calibration"]["reviewed"])
                self.assertGreater(len(app.data("run_test")["pitch_lines"]), 10)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
