import unittest

from footballer_analyzer.identity import validate_bindings
from footballer_analyzer.metrics import calculate_motion, summarize


def manifest():
    return {"run_id": "run_test", "start_frame": 0, "end_frame": 10, "video": {"fps": 25}, "config": {"stride": 1, "speed_limit_mps": 12, "min_cutoff_hz": 1, "filter_beta": 0, "match_id": "match", "clip_id": "clip"}}


def calibration():
    return {"homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "roi": [[0, 0], [100, 0], [100, 60], [0, 60]], "height": 100, "start_frame": 0, "end_frame": 10, "pitch_length_m": 105, "pitch_width_m": 68}


def binding(track_id=1, player_id="player", start=0, end=10):
    return {"track_id": track_id, "player_id": player_id, "start_frame": start, "end_frame": end, "team_id": "home", "action": "bind"}


def observation(frame_index, horizontal, track_id=1):
    return {"frame_index": frame_index, "timestamp_s": frame_index / 25, "track_id": track_id, "bbox_x1": horizontal - 1, "bbox_x2": horizontal + 1, "bbox_y2": 20, "team_id": "unknown"}


class MetricsTests(unittest.TestCase):
    def test_missing_observation_starts_new_segment_without_bridging(self):
        bindings = [binding()]
        rows = list(calculate_motion([observation(0, 10), observation(1, 10.1), observation(3, 15), observation(4, 15.1)], manifest(), calibration(), bindings))
        self.assertIsNone(rows[2]["distance_delta_m"])
        self.assertNotEqual(rows[1]["segment_id"], rows[2]["segment_id"])
        summary = summarize(rows, manifest(), bindings)[0]
        self.assertAlmostEqual(summary["valid_seconds"], 0.08)
        self.assertAlmostEqual(summary["distance_m"], sum(row["distance_delta_m"] or 0 for row in rows))
        self.assertAlmostEqual(summary["avg_speed_mps"], summary["distance_m"] / 0.08)

    def test_identity_change_splits_track(self):
        rows = list(calculate_motion([observation(0, 10), observation(1, 10.1)], manifest(), calibration(), [binding(end=1), binding(player_id="other", start=1)]))
        self.assertIsNone(rows[1]["distance_delta_m"])
        self.assertEqual(rows[1]["player_id"], "other")

    def test_unbound_and_jump_cannot_add_distance(self):
        rows = list(calculate_motion([observation(0, 10), observation(1, 20), observation(2, 20.1), observation(3, 21, 2)], manifest(), calibration(), [binding()]))
        self.assertEqual(rows[1]["reason"], "speed_jump")
        self.assertIsNone(rows[2]["distance_delta_m"])
        self.assertEqual(rows[3]["reason"], "unbound")
        self.assertIsNone(summarize(rows, manifest(), [binding()])[0]["distance_m"])

    def test_rejects_overlapping_player_tracks(self):
        payload = {"run_id": "run_test", "bindings": [binding(), binding(track_id=2)]}
        with self.assertRaisesRegex(ValueError, "同时"):
            validate_bindings(payload, manifest(), {"player": {}}, [{"track_id": 1}, {"track_id": 2}])


if __name__ == "__main__":
    unittest.main()
