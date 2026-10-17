"""验证 COCO 足球检测与球员产物隔离。"""

from contextlib import ExitStack
from pathlib import Path
import unittest
from unittest.mock import call, patch

import numpy as np
import supervision as sv

from footballer_analyzer.config import Config, config_from_dict
from footballer_analyzer.detection import YoloDetector
from footballer_analyzer.pipeline import track


def detections(boxes, scores, classes):
    """构造指定框、置信度和 COCO 类别的检测结果。"""
    return sv.Detections(
        xyxy=np.asarray(boxes, dtype=float).reshape(-1, 4),
        confidence=np.asarray(scores, dtype=float),
        class_id=np.asarray(classes, dtype=int),
    )


class BallDetectionTests(unittest.TestCase):
    """验证检测器接口、配置和完整管线产物。"""

    def test_single_predict_splits_classes_and_applies_independent_thresholds(self):
        """单次 predict 返回过滤后的人物和 32 类足球候选。"""
        model_result = object()
        model_detections = detections(
            [[0, 0, 10, 20], [1, 1, 5, 5], [10, 10, 14, 14], [20, 20, 24, 24]],
            [0.6, 0.08, 0.2, 0.14],
            [0, 0, 32, 32],
        )
        with patch("footballer_analyzer.detection.YOLO") as model_type, patch(
            "footballer_analyzer.detection.sv.Detections.from_ultralytics", return_value=model_detections
        ):
            model_type.return_value.names = {0: "person", 32: "sports ball"}
            model_type.return_value.predict.return_value = [model_result]
            weights = type("Weights", (), {"is_file": lambda self: True, "__str__": lambda self: "weights.pt"})()
            detector = YoloDetector(weights, confidence=0.1, ball_confidence=0.15)
            people, balls = detector.detect_objects(np.zeros((32, 32, 3), dtype=np.uint8))

        self.assertEqual(len(model_type.return_value.predict.call_args_list), 1)
        kwargs = model_type.return_value.predict.call_args.kwargs
        self.assertEqual(kwargs["classes"], [0, 32])
        self.assertEqual(kwargs["conf"], 0.1)
        self.assertEqual(people.class_id.tolist(), [0])
        self.assertEqual(people.confidence.tolist(), [0.6])
        self.assertEqual(balls.class_id.tolist(), [32])
        self.assertEqual(balls.confidence.tolist(), [0.2])

    def test_detect_remains_person_only_and_model_names_are_validated(self):
        """旧 detect 保持仅人物语义，自定义类别映射会快速失败。"""
        person_result = object()
        people = detections([[0, 0, 10, 20]], [0.9], [0])
        with patch("footballer_analyzer.detection.YOLO") as model_type, patch(
            "footballer_analyzer.detection.sv.Detections.from_ultralytics", return_value=people
        ):
            model_type.return_value.names = {0: "person", 32: "sports ball"}
            model_type.return_value.predict.return_value = [person_result]
            weights = type("Weights", (), {"is_file": lambda self: True, "__str__": lambda self: "weights.pt"})()
            detector = YoloDetector(weights)
            self.assertEqual(detector.detect(np.zeros((16, 16, 3), dtype=np.uint8)).class_id.tolist(), [0])
            self.assertEqual(model_type.return_value.predict.call_args.kwargs["classes"], [0])

            model_type.return_value.names = {0: "athlete", 32: "sports ball"}
            with self.assertRaisesRegex(ValueError, "类别不兼容"):
                detector.detect_objects(np.zeros((16, 16, 3), dtype=np.uint8))

    def test_disabled_ball_detection_preserves_legacy_detect_api_and_skips_class_32(self):
        """禁用时只推理 0 类，并返回空足球检测。"""
        people = detections([[0, 0, 8, 12]], [0.9], [0])
        with patch("footballer_analyzer.detection.YOLO") as model_type, patch(
            "footballer_analyzer.detection.sv.Detections.from_ultralytics", return_value=people
        ):
            model_type.return_value.names = {0: "person"}
            model_type.return_value.predict.return_value = [object()]
            weights = type("Weights", (), {"is_file": lambda self: True, "__str__": lambda self: "weights.pt"})()
            detector = YoloDetector(weights, ball_detection_enabled=False)
            person_only = detector.detect(np.zeros((16, 16, 3), dtype=np.uint8))
            objects_people, balls = detector.detect_objects(np.zeros((16, 16, 3), dtype=np.uint8))

        self.assertEqual(person_only.class_id.tolist(), [0])
        self.assertEqual(objects_people.class_id.tolist(), [0])
        self.assertEqual(len(balls), 0)
        self.assertEqual(model_type.return_value.predict.call_args.kwargs["classes"], [0])

    def test_config_validates_ball_controls_and_defaults_to_enabled(self):
        """足球检测默认启用，并独立校验置信度配置。"""
        payload = {"video": "video.mp4", "weights": "model.pt", "match_id": "test", "clip_id": "test"}
        config = config_from_dict(payload)
        self.assertTrue(config.ball_detection_enabled)
        self.assertEqual(config.ball_confidence, 0.15)
        for key, invalid in [("ball_confidence", 0), ("ball_confidence", 1), ("ball_confidence", float("nan")), ("ball_detection_enabled", "yes")]:
            with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                config_from_dict({**payload, key: invalid})

    def test_pipeline_writes_ball_candidates_separately_and_only_annotates_preview(self):
        """足球单独写行并标注，不影响球队或球员产物。"""
        player = detections([[5, 3, 30, 46]], [0.9], [0])
        ball = detections([[36, 18, 42, 24]], [0.8], [32])
        events = []
        output_files = {}
        video_path = Path("source.avi")
        weights = Path("weights.pt")
        output_root = Path("out")
        info = type("Info", (), {"fps": 25.0, "frame_count": 4, "width": 64, "height": 48, "snapshot": lambda self: {"frame_count": 4}})()
        packets = [type("Packet", (), {"frame_index": i, "timestamp_s": i / 25, "image": np.zeros((48, 64, 3), np.uint8)})() for i in range(4)]
        preview = type("Preview", (), {"isOpened": lambda self: True, "write": lambda self, frame: events.append(("preview", frame)), "release": lambda self: None})()
        class Writer:
            def __init__(self, path, schema, batch_size=2048):
                self.path = path
                self.rows = []
                self.key = path.name
            def __enter__(self):
                return self
            def append(self, row):
                self.rows.append(row)
            def __exit__(self, error_type, error, traceback):
                output_files[self.key] = self.rows

        run_dir = Path("out/run_test")
        config = Config(video_path, weights, output_root, Path("players.yaml"), "test", "test")
        with ExitStack() as stack:
            stack.enter_context(patch("footballer_analyzer.pipeline.inspect_video", return_value=info))
            stack.enter_context(patch("footballer_analyzer.pipeline.file_hash", return_value="hash"))
            stack.enter_context(patch("footballer_analyzer.pipeline.new_id", return_value="run_test"))
            stack.enter_context(patch("footballer_analyzer.pipeline.write_json", side_effect=lambda path, payload: events.append(("json", path.name, payload))))
            stack.enter_context(patch("footballer_analyzer.pipeline.write_yaml"))
            stack.enter_context(patch("footballer_analyzer.pipeline.versions", return_value={}))
            stack.enter_context(patch("footballer_analyzer.pipeline.read_frames", side_effect=[iter(packets[:1]), iter(packets)]))
            detector_type = stack.enter_context(patch("footballer_analyzer.pipeline.YoloDetector"))
            tracker_type = stack.enter_context(patch("footballer_analyzer.pipeline.ByteTracker"))
            teams = stack.enter_context(patch("footballer_analyzer.pipeline.TeamClassifier")).return_value
            stack.enter_context(patch("footballer_analyzer.pipeline.RowWriter", Writer))
            stack.enter_context(patch("footballer_analyzer.pipeline.cv2.VideoWriter", return_value=preview))
            stack.enter_context(patch("footballer_analyzer.pipeline.cv2.VideoWriter_fourcc", return_value=0))
            stack.enter_context(patch("footballer_analyzer.pipeline.cv2.rectangle"))
            put_text = stack.enter_context(patch("footballer_analyzer.pipeline.cv2.putText"))
            stack.enter_context(patch("footballer_analyzer.pipeline.cv2.imwrite", return_value=True))
            stack.enter_context(patch.object(Path, "is_file", return_value=True))
            stack.enter_context(patch.object(Path, "mkdir"))
            stack.enter_context(patch.object(Path, "write_text"))
            detector_type.return_value.detect.return_value = player
            detector_type.return_value.detect_objects.side_effect = [(player, ball)] * 4
            tracked = detections(player.xyxy, player.confidence, player.class_id)
            tracked.tracker_id = np.array([1])
            tracker_type.return_value.update.return_value = tracked
            teams.model = None
            teams.predict.return_value = "cluster_0"

            self.assertEqual(track(config), run_dir)

        player_rows = output_files["detections.parquet"]
        ball_rows = output_files["ball_detections.parquet"]
        track_rows = output_files["tracks.parquet"]
        self.assertEqual(len(player_rows), 4)
        self.assertTrue(all(row["class_id"] == 0 for row in player_rows))
        self.assertEqual(len(ball_rows), 4)
        self.assertTrue(all(row["class_id"] == 32 for row in ball_rows))
        self.assertEqual(len(track_rows), 4)
        self.assertTrue(all(row["class_id"] == 0 for row in track_rows))
        self.assertEqual(teams.predict.call_count, 4)
        tracker_type.return_value.update.assert_called()
        tracker_type.return_value.update.assert_has_calls([call(player, packet.timestamp_s) for packet in packets])
        self.assertTrue(any(call.args[1].startswith("ball ") for call in put_text.call_args_list))
        self.assertEqual(len(detector_type.return_value.detect_objects.call_args_list), 4)
        manifest_updates = [event[2] for event in events if len(event) > 2 and event[1] == "manifest.json"]
        complete_manifest = next(payload for payload in manifest_updates if payload.get("status") == "complete")
        self.assertEqual(complete_manifest["ball_detection_count"], 4)


if __name__ == "__main__":
    unittest.main()
