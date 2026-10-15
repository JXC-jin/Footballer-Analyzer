"""验证白线自动标定在平移、变焦、遮挡与失效恢复时的几何正确性。"""

import unittest
from pathlib import Path
import tempfile

import cv2
import numpy as np

from footballer_analyzer.dynamic_calibration import WhiteLineCalibrator
from footballer_analyzer.calibration import load_calibration, project_observation
from footballer_analyzer.pitch_geometry import pitch_polylines, transform_points
from footballer_analyzer.storage import read_json
from footballer_analyzer.white_line_pipeline import create_line_run


def synthetic_pitch(shift=0, zoom=1):
    """生成已知单应性的战术视角草地图与白线，返回 BGR 图像和米到像素矩阵。"""
    source = np.array([[0, 0], [105, 0], [105, 68], [0, 68]], np.float32)
    target = np.array([[-200, 130], [1160, 130], [1800, 480], [-840, 480]], np.float32)
    target = (target - [480, 270]) * zoom + [480 + shift, 270]
    matrix = cv2.getPerspectiveTransform(source, target.astype(np.float32))
    image = np.full((540, 960, 3), (45, 115, 55), np.uint8)
    image[:100] = (35, 35, 35)
    cv2.putText(image, "WHITE ADVERTISING", (100, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    for _, points in pitch_polylines(spacing=0.2):
        pixels = transform_points(points, matrix)
        cv2.polylines(image, [np.rint(pixels).astype(np.int32)], False, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.rectangle(image, (600, 250), (609, 277), (240, 240, 240), -1)
    return image, matrix


class DynamicCalibrationTests(unittest.TestCase):
    """用独立已知投影验证自动初始化、逐帧更新与错误拒绝。"""

    def test_moving_camera_updates_matrix_without_direction_input(self):
        """移动和变焦时，每帧重新拟合；独立场地点的反投影误差应小于六像素。"""
        calibrator = WhiteLineCalibrator()
        checkpoints = np.array([[45, 15], [60, 50], [52.5, 34]], float)
        first_matrix = None
        for frame_index in range(8):
            image, truth = synthetic_pitch(shift=frame_index * 5, zoom=1 + frame_index * 0.006)
            result, _ = calibrator.process(image, frame_index)
            self.assertEqual(result["status"], "valid", result)
            predicted = transform_points(checkpoints, np.linalg.inv(result["homography"]))
            expected = transform_points(checkpoints, truth)
            self.assertLess(float(np.max(np.linalg.norm(predicted - expected, axis=1))), 6)
            if frame_index == 0:
                first_matrix = result["homography"]
        self.assertFalse(np.allclose(first_matrix, result["homography"]))

    def test_missing_lines_invalidates_old_matrix_and_can_recover(self):
        """没有白线时不沿用旧 H，恢复场地画面后重新自动初始化。"""
        calibrator = WhiteLineCalibrator()
        image, _ = synthetic_pitch()
        self.assertEqual(calibrator.process(image, 0)[0]["status"], "valid")
        blank = np.full_like(image, (45, 115, 55))
        failed, _ = calibrator.process(blank, 1)
        self.assertIsNone(failed["homography"])
        self.assertIsNone(calibrator.previous)
        shifted, _ = synthetic_pitch(shift=-90, zoom=1.05)
        self.assertEqual(calibrator.process(shifted, 2)[0]["status"], "valid")

    def test_one_line_and_advertising_cannot_calibrate(self):
        """单根白线和场外白色广告不足以确定球场映射，必须返回无效。"""
        image = np.full((540, 960, 3), (45, 115, 55), np.uint8)
        image[:110] = (25, 25, 25)
        cv2.putText(image, "WHITE ADVERTISING", (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
        cv2.line(image, (0, 350), (959, 350), (255, 255, 255), 3)
        self.assertIsNone(WhiteLineCalibrator().process(image, 0)[0]["homography"])

    def test_nonstandard_dimensions_are_rejected(self):
        """标准十一人制模型不接受非有限尺寸或小场尺寸。"""
        for length, width in [(float("nan"), 68), (105, 0), (50, 30)]:
            with self.assertRaises(ValueError):
                WhiteLineCalibrator(length, width)

    def test_projection_requires_exact_frame_and_never_reuses_failed_mapping(self):
        """不同源帧使用各自 H；失败和缺帧均不借用相邻帧的映射。"""
        roi = [[0, 0], [99, 0], [99, 99], [0, 99]]
        calibration = {"mode": "white_lines", "start_frame": 0, "end_frame": 4, "height": 100, "pitch_length_m": 105, "pitch_width_m": 68, "frames": {
            "0": {"status": "valid", "homography": np.eye(3).tolist(), "roi": roi},
            "1": {"status": "valid", "homography": [[1, 0, -5], [0, 1, 0], [0, 0, 1]], "roi": roi},
            "2": {"status": "insufficient_geometry", "homography": None},
        }}
        observation = {"bbox_x1": 8, "bbox_x2": 12, "bbox_y2": 20}
        self.assertEqual(project_observation({**observation, "frame_index": 0}, calibration), (10, 20, None))
        self.assertEqual(project_observation({**observation, "frame_index": 1}, calibration), (5, 20, None))
        for frame_index in [2, 3]:
            self.assertEqual(project_observation({**observation, "frame_index": frame_index}, calibration), (None, None, "calibration_unreliable"))
        self.assertEqual(project_observation({**observation, "frame_index": 4}, calibration)[2], "outside_calibration_window")

    def test_independent_review_run_persists_all_frames_and_invalid_evidence(self):
        """独立验收运行无需人物模型，保存每帧映射、失败记录和透明白线掩膜。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "moving.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (960, 540))
            for frame_index in range(3):
                image, _ = synthetic_pitch(shift=frame_index * 4)
                writer.write(image if frame_index != 1 else np.full_like(image, (45, 115, 55)))
            writer.release()
            run_dir = create_line_run(video, root / "outputs")
            calibration = load_calibration(run_dir)
            self.assertEqual(calibration["summary"]["frame_count"], 3)
            self.assertEqual(calibration["summary"]["valid_frames"], 2)
            self.assertIsNone(calibration["frames"]["1"]["homography"])
            self.assertFalse(calibration["reviewed"])
            self.assertEqual(read_json(run_dir / "manifest.json")["status"], "complete")
            evidence = cv2.imread(str(run_dir / "calibrations" / calibration["calibration_id"] / "masks/1.png"), cv2.IMREAD_UNCHANGED)
            self.assertEqual(evidence.shape[2], 4)


if __name__ == "__main__":
    unittest.main()
