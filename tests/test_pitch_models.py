"""验证独立场地模板、米制标线与稳定参考点。"""

import unittest

import numpy as np

from footballer_analyzer.pitch_geometry import pitch_polylines
from footballer_analyzer.pitch_models import pitch_landmarks, pitch_spec


class PitchModelTests(unittest.TestCase):
    """检查八人制参数不会借用十一人制几何。"""

    def test_independent_eight_geometry_and_landmarks(self):
        """八人制禁区、球门区及中圈符合 JFA 预设。"""
        spec = pitch_spec("eight")
        self.assertEqual((spec["length_m"], spec["width_m"]), (68, 50))
        self.assertEqual((spec["penalty_width_m"], spec["goal_area_width_m"]), (29, 13))
        points = {point["id"]: point["pitch"] for point in pitch_landmarks(spec)}
        self.assertEqual(points["left_penalty_far_front"], [12, 10.5])
        self.assertEqual(points["right_goal_area_near_front"], [64, 31.5])
        self.assertEqual(points["left_penalty_mark"], [8, 25])
        curves = dict(pitch_polylines(model="eight"))
        np.testing.assert_allclose(curves["left_penalty_front"][[0, -1]], [[12, 10.5], [12, 39.5]])
        self.assertAlmostEqual(np.linalg.norm(curves["center_circle"][0] - [34, 25]), 7)
        self.assertEqual(len(points), len(pitch_landmarks(spec)))

    def test_eleven_defaults_and_dimension_changes(self):
        """保留十一人制默认几何，调整整体尺寸不缩放禁区。"""
        curves = dict(pitch_polylines())
        np.testing.assert_allclose(curves["left_penalty_front"][[0, -1]], [[16.5, 13.84], [16.5, 54.16]])
        self.assertEqual(pitch_spec("eight", 70, 52)["penalty_width_m"], 29)
        for arguments in [("other",), ("eight", float("nan")), ("eight", 20, 50), ("eight", 68, 25)]:
            with self.assertRaises(ValueError):
                pitch_spec(*arguments)


if __name__ == "__main__":
    unittest.main()
