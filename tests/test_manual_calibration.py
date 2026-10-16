"""验证具名参考点标定、全段投影与预览接口的保存边界。"""

import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

import cv2
import numpy as np

from footballer_analyzer.calibration import build_calibration, load_calibration, project_observation
from footballer_analyzer.pitch_geometry import transform_points
from footballer_analyzer.pitch_models import pitch_landmarks, pitch_spec
from footballer_analyzer.server import ReviewApp, handler_for
from footballer_analyzer.storage import write_json


class ManualCalibrationTests(unittest.TestCase):
    """用独立已知单应性检查模型解析、失效拒绝及不落盘预览。"""

    def fixture(self):
        """返回八人制单侧禁区六点、运行清单与真实米到像素矩阵。"""
        manifest = {"video_sha256": "test", "video": {"width": 1280, "height": 720}, "start_frame": 0, "end_frame": 300, "config": {"pitch_model": "eight", "pitch_length_m": 68, "pitch_width_m": 50}}
        truth = np.array([[12, 3, 100], [1, 8, 80], [0.002, 0.003, 1]], float)
        landmarks = {point["id"]: point["pitch"] for point in pitch_landmarks(pitch_spec("eight"))}
        identifiers = ["left_penalty_far_back", "left_penalty_far_front", "left_penalty_near_back", "left_penalty_near_front", "left_goal_area_far_front", "left_goal_area_near_front"]
        images = transform_points([landmarks[identifier] for identifier in identifiers], truth)
        payload = {"pitch_model": "eight", "reference_frame": 147, "point_pairs": [{"landmark_id": identifier, "image": image.tolist()} for identifier, image in zip(identifiers, images)]}
        return manifest, payload, truth

    def test_known_mapping_projects_beyond_reference_hull_for_whole_run(self):
        """禁区内标定后，禁区外独立脚点在首、中、末帧均准确投影。"""
        manifest, payload, truth = self.fixture()
        calibration = build_calibration(manifest, payload)
        pixel = transform_points([[40, 25]], truth)[0]
        for frame in [0, 147, 299]:
            result = project_observation({"frame_index": frame, "bbox_x1": pixel[0] - 4, "bbox_x2": pixel[0] + 4, "bbox_y2": pixel[1]}, calibration)
            np.testing.assert_allclose(result[:2], [40, 25], atol=1e-4)
            self.assertIsNone(result[2])
        self.assertEqual(calibration["pitch_spec"]["penalty_width_m"], 29)
        self.assertEqual(calibration["mode"], "static")
        self.assertEqual(calibration["point_pairs"], payload["point_pairs"])
        self.assertLess(calibration["fit_rmse_px"], 1e-3)

    def test_rejects_invalid_duplicate_collinear_and_degenerate_points(self):
        """无效参考点、重复点、越界图像点及退化点集不可保存。"""
        manifest, payload, _ = self.fixture()
        invalid_pairs = [
            [{"landmark_id": "not-a-point", "image": [10, 10]}] * 4,
            [payload["point_pairs"][0]] * 4,
            [{**point, "image": [9999, 0]} for point in payload["point_pairs"]],
            [{**point, "image": [index * 5, 10]} for index, point in enumerate(payload["point_pairs"])],
            [{**point, "image": [float("nan"), 10]} for point in payload["point_pairs"]],
        ]
        for pairs in invalid_pairs:
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                build_calibration(manifest, {**payload, "point_pairs": pairs})

    def test_preview_does_not_write_and_save_restores_named_pairs(self):
        """HTTP 预览不写文件，保存后可读取模型及参考帧对应关系。"""
        manifest, payload, _ = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run_test"
            write_json(run_dir / "manifest.json", manifest)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(ReviewApp(temporary)))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base + "/api/pitch-models?model=eight&length=70&width=52") as response:
                    definition = json.load(response)["models"][0]
                    self.assertEqual(definition["spec"]["length_m"], 70)
                    self.assertTrue(definition["landmarks"])
                for action in ["calibration-preview", "calibration"]:
                    request = Request(base + "/api/runs/run_test/" + action, data=json.dumps({**payload, "reviewed": True}).encode(), headers={"Content-Type": "application/json"})
                    with urlopen(request) as response:
                        result = json.load(response)
                        self.assertEqual(result["reviewed"], action == "calibration")
                    self.assertEqual((run_dir / "calibration.yaml").exists(), action == "calibration")
                restored = load_calibration(run_dir)
                self.assertEqual(restored["point_pairs"], payload["point_pairs"])
                self.assertEqual(restored["reference_frame"], 147)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
