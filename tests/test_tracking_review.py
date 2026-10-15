"""验证人工参考框的一对一匹配及检测、关联丢失的区分。"""

import unittest

from scripts.review_tracking import evaluate_frames, match_boxes


def observation(box, track_id=1, frame_index=0):
    """构造用于抽样复查的像素框与轨迹行。"""
    return dict(zip(["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"], box), track_id=track_id, frame_index=frame_index)


class TrackingReviewTests(unittest.TestCase):
    """确保抽样指标不能用一个合并框冒充多个可见球员。"""

    def test_one_detection_cannot_count_as_two_players(self):
        """两个参考框与同一候选重叠时最多命中一人，远处候选不匹配。"""
        labels = {"first": [0, 0, 10, 20], "second": [2, 0, 12, 20]}
        candidates = [observation([0, 0, 10, 20]), observation([100, 100, 110, 120])]
        self.assertEqual(len(match_boxes(labels, candidates)), 1)

    def test_detector_miss_and_association_miss_are_separate(self):
        """分别计入无检测、检测未关联和抽样编号变化；旧运行漏检信息为未知。"""
        labels = {"0": {"keeper": [0, 0, 10, 20], "white": [20, 0, 30, 20], "lime": [40, 0, 50, 20]}, "5": {"keeper": [0, 0, 10, 20]}}
        detections = [observation([0, 0, 10, 20]), observation([20, 0, 30, 20]), observation([0, 0, 10, 20], frame_index=5)]
        tracks = [observation([0, 0, 10, 20]), observation([0, 0, 10, 20], track_id=2, frame_index=5)]
        result = evaluate_frames(labels, detections, tracks)
        self.assertEqual((result["expected"], result["detected"], result["tracked"]), (4, 3, 2))
        self.assertEqual(result["frames"][0]["detection_misses"], ["lime"])
        self.assertEqual(result["frames"][0]["association_misses"], ["white"])
        self.assertEqual(result["sampled_id_changes"], 1)
        legacy = evaluate_frames(labels, None, tracks)
        self.assertIsNone(legacy["detected"])
        self.assertIsNone(legacy["frames"][0]["detection_misses"])


if __name__ == "__main__":
    unittest.main()
