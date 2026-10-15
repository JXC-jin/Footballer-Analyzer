from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from footballer_analyzer.calibration import build_calibration, save_calibration
from footballer_analyzer.config import Config
from footballer_analyzer.reporting import create_report, history
from footballer_analyzer.storage import TRACK_SCHEMA, RowWriter, file_hash, read_json, read_rows, read_yaml, write_json, write_yaml


class ReportingTests(unittest.TestCase):
    def prepare_run(self, temporary):
        root = Path(temporary)
        video_path = root / "source.avi"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
        for _ in range(10):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        writer.release()
        players_path = root / "players.yaml"
        write_yaml(players_path, {"players": [{"player_id": "player_001", "display_name": "Synthetic test player"}]})
        config = Config(video_path, root / "unused.pt", root, players_path, "synthetic_match", "synthetic_clip")
        run_dir = root / "run_test"
        run_dir.mkdir()
        manifest = {"run_id": run_dir.name, "status": "complete", "config": config.snapshot(), "video": {"width": 64, "height": 48, "fps": 25.0, "frame_count": 10}, "video_sha256": file_hash(video_path), "start_frame": 0, "end_frame": 10}
        write_json(run_dir / "manifest.json", manifest)
        with RowWriter(run_dir / "tracks.parquet", TRACK_SCHEMA) as rows:
            for frame_index in range(10):
                rows.append({"frame_index": frame_index, "timestamp_s": frame_index / 25, "track_id": 1, "bbox_x1": 8 + frame_index * 0.1, "bbox_y1": 3, "bbox_x2": 10 + frame_index * 0.1, "bbox_y2": 20, "confidence": 0.9, "class_id": 0, "team_id": "cluster_0"})
        write_json(run_dir / "catalog.json", [{"track_id": 1, "start_frame": 0, "end_frame": 10}])
        write_yaml(run_dir / "bindings.yaml", {"run_id": run_dir.name, "bindings": [{"track_id": 1, "start_frame": 0, "end_frame": 10, "player_id": "player_001", "team_id": "home", "action": "bind"}]})
        points = {"image_points": [[0, 0], [63, 0], [63, 47], [0, 47]], "pitch_points": [[0, 0], [6.3, 0], [6.3, 4.7], [0, 4.7]], "pitch_length_m": 10, "pitch_width_m": 7, "reviewed": True}
        save_calibration(run_dir, points)
        return run_dir, manifest

    def test_report_recompute_preserves_tracks_and_rejects_duplicate_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, manifest = self.prepare_run(temporary)
            original_hash = file_hash(run_dir / "tracks.parquet")
            first = create_report(run_dir)
            second = create_report(run_dir)
            self.assertNotEqual(first, second)
            self.assertEqual(file_hash(run_dir / "tracks.parquet"), original_hash)
            self.assertEqual(read_json(second / "manifest.json")["status"], "complete")
            self.assertTrue((second / "report.html").is_file())
            capture = cv2.VideoCapture(str(second / "review.mp4"))
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 10)
            capture.release()
            summary = list(read_rows(second / "summary.parquet"))[0]
            motions = list(read_rows(second / "motion.parquet"))
            self.assertAlmostEqual(summary["distance_m"], sum(row["distance_delta_m"] or 0 for row in motions))
            self.assertAlmostEqual(summary["valid_seconds"], 9 / 25)
            self.assertEqual(history("player_001", [second])["records"][0]["clip_id"], "synthetic_clip")
            with self.assertRaisesRegex(ValueError, "重复"):
                history("player_001", [first, second])

    def test_draft_calibration_prevents_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, manifest = self.prepare_run(temporary)
            calibration = read_yaml(run_dir / "calibration.yaml")
            calibration["reviewed"] = False
            write_yaml(run_dir / "calibration.yaml", calibration)
            with self.assertRaisesRegex(ValueError, "草稿"):
                create_report(run_dir)

    def test_identity_correction_changes_report_without_changing_tracks(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, manifest = self.prepare_run(temporary)
            original_hash = file_hash(run_dir / "tracks.parquet")
            first = create_report(run_dir)
            players_path = Path(manifest["config"]["players"])
            write_yaml(players_path, {"players": [
                {"player_id": "player_001", "display_name": "First synthetic player"},
                {"player_id": "player_002", "display_name": "Second synthetic player"},
            ]})
            payload = read_yaml(run_dir / "bindings.yaml")
            original = payload["bindings"][0]
            payload["bindings"] = [
                {**original, "end_frame": 5},
                {**original, "start_frame": 5, "player_id": "player_002"},
            ]
            write_yaml(run_dir / "bindings.yaml", payload)
            second = create_report(run_dir)
            self.assertEqual(file_hash(run_dir / "tracks.parquet"), original_hash)
            self.assertEqual(len(read_yaml(first / "bindings.yaml")["bindings"]), 1)
            summary = list(read_rows(second / "summary.parquet"))
            self.assertEqual({row["player_id"] for row in summary}, {"player_001", "player_002"})
            self.assertTrue(all(abs(row["valid_seconds"] - 4 / 25) < 1e-8 for row in summary))
            boundary = next(row for row in read_rows(second / "motion.parquet") if row["frame_index"] == 5)
            self.assertIsNone(boundary["distance_delta_m"])
            self.assertIsNone(boundary["speed_mps"])

    def test_collinear_calibration_points_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "共线"):
            build_calibration({"video": {"width": 100, "height": 100}}, {"image_points": [[1, 1], [2, 2], [3, 3], [4, 4]], "pitch_points": [[0, 0], [10, 0], [10, 10], [0, 10]]})


if __name__ == "__main__":
    unittest.main()
