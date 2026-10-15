"""验证身份边界、缺失和异常跳变的切段规则，以及距离汇总和绑定冲突。"""

import unittest

from footballer_analyzer.identity import validate_bindings
from footballer_analyzer.metrics import calculate_motion, summarize


def manifest():
    """构造 25 FPS、10 帧、stride=1 的统计测试清单。"""
    return {"run_id": "run_test", "start_frame": 0, "end_frame": 10, "video": {"fps": 25}, "config": {"stride": 1, "speed_limit_mps": 12, "min_cutoff_hz": 1, "filter_beta": 0, "match_id": "match", "clip_id": "clip"}}


def calibration():
    """返回单位单应性、有效矩形区域与测试窗口的标定字典。"""
    return {"homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "roi": [[0, 0], [100, 0], [100, 60], [0, 60]], "height": 100, "start_frame": 0, "end_frame": 10, "pitch_length_m": 105, "pitch_width_m": 68}


def binding(track_id=1, player_id="player", start=0, end=10):
    """构造主队球员在指定左闭右开源帧区间内的身份绑定。"""
    return {"track_id": track_id, "player_id": player_id, "start_frame": start, "end_frame": end, "team_id": "home", "action": "bind"}


def observation(frame_index, horizontal, track_id=1):
    """构造指定源帧和横向脚点的观测，时间按 25 FPS 换算。"""
    return {"frame_index": frame_index, "timestamp_s": frame_index / 25, "track_id": track_id, "bbox_x1": horizontal - 1, "bbox_x2": horizontal + 1, "bbox_y2": 20, "team_id": "unknown"}


class MetricsTests(unittest.TestCase):
    """检查统计只累加同一身份、同一连续段内的有效时间边。"""

    def test_missing_observation_starts_new_segment_without_bridging(self):
        """漏掉采样帧时新建轨迹段，缺口不贡献距离或有效时间。"""
        bindings = [binding()]
        rows = list(calculate_motion([observation(0, 10), observation(1, 10.1), observation(3, 15), observation(4, 15.1)], manifest(), calibration(), bindings))
        self.assertIsNone(rows[2]["distance_delta_m"])
        self.assertNotEqual(rows[1]["segment_id"], rows[2]["segment_id"])
        summary = summarize(rows, manifest(), bindings)[0]
        self.assertAlmostEqual(summary["valid_seconds"], 0.08)
        self.assertAlmostEqual(summary["distance_m"], sum(row["distance_delta_m"] or 0 for row in rows))
        self.assertAlmostEqual(summary["avg_speed_mps"], summary["distance_m"] / 0.08)

    def test_identity_change_splits_track(self):
        """同一轨迹切换球员身份时，边界点不连接上一名球员。"""
        rows = list(calculate_motion([observation(0, 10), observation(1, 10.1)], manifest(), calibration(), [binding(end=1), binding(player_id="other", start=1)]))
        self.assertIsNone(rows[1]["distance_delta_m"])
        self.assertEqual(rows[1]["player_id"], "other")

    def test_fragmented_tracks_share_identity_without_bridging(self):
        """同一球员绑定断开的两条轨迹时，各段分别计距后合并汇总。"""
        bindings = [binding(end=2), binding(track_id=2, start=3)]
        rows = list(calculate_motion([
            observation(0, 10), observation(1, 10.1),
            observation(3, 15, 2), observation(4, 15.1, 2),
        ], manifest(), calibration(), bindings))
        self.assertEqual({row["player_id"] for row in rows}, {"player"})
        self.assertIsNone(rows[2]["distance_delta_m"])
        self.assertNotEqual(rows[1]["segment_id"], rows[2]["segment_id"])
        summary = summarize(rows, manifest(), bindings)
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary[0]["valid_seconds"], 2 / 25)

    def test_unbound_and_jump_cannot_add_distance(self):
        """未绑定和异常跳变不计距；没有有效时间时距离显示为数据不足。"""
        rows = list(calculate_motion([observation(0, 10), observation(1, 20), observation(2, 20.1), observation(3, 21, 2)], manifest(), calibration(), [binding()]))
        self.assertEqual(rows[1]["reason"], "speed_jump")
        self.assertIsNone(rows[2]["distance_delta_m"])
        self.assertEqual(rows[3]["reason"], "unbound")
        self.assertIsNone(summarize(rows, manifest(), [binding()])[0]["distance_m"])

    def test_rejects_overlapping_player_tracks(self):
        """拒绝同一球员在重叠时间绑定两条轨迹。"""
        payload = {"run_id": "run_test", "bindings": [binding(), binding(track_id=2)]}
        with self.assertRaisesRegex(ValueError, "同时"):
            validate_bindings(payload, manifest(), {"player": {}}, [{"track_id": 1}, {"track_id": 2}])


if __name__ == "__main__":
    unittest.main()
