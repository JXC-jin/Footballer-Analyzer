from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from footballer_analyzer.video import read_frames


class VideoTests(unittest.TestCase):
    def test_stride_keeps_source_frame_numbers_and_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            video_path = Path(temporary) / "source.avi"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
            for frame_index in range(12):
                writer.write(np.full((48, 64, 3), frame_index * 10, dtype=np.uint8))
            writer.release()
            packets = list(read_frames(video_path, start_frame=2, end_frame=11, stride=3))
            self.assertEqual([packet.frame_index for packet in packets], [2, 5, 8])
            self.assertEqual([packet.timestamp_s for packet in packets], [0.08, 0.2, 0.32])
            self.assertLess(abs(float(packets[1].image.mean()) - 50), 3)


if __name__ == "__main__":
    unittest.main()
