"""逐帧提取白线、自动匹配场地几何并保存独立映射；失败帧不复用旧坐标。"""

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.optimize import least_squares

from .pitch_geometry import circle_homographies, detect_long_lines, extract_white_lines, pitch_polylines, rectangle_homographies, transform_points


class WhiteLineCalibrator:
    """维护上帧几何先验，每帧重新检测并拟合白线；不依赖人物 ID 或人工方向。"""

    def __init__(self, length=105.0, width=68.0, work_width=960):
        """设置米制尺寸与工作分辨率，建立标准场地采样。"""
        if not np.isfinite([length, width]).all() or not 90 <= length <= 120 or not 60 <= width <= 90:
            raise ValueError("自动白线标定要求标准十一人制球场：长 90–120 m、宽 60–90 m")
        self.length, self.width, self.work_width = float(length), float(width), work_width
        self.curves = pitch_polylines(length, width)
        self.model = np.concatenate([points for _, points in self.curves])
        self.groups = np.concatenate([np.full(len(points), index) for index, (_, points) in enumerate(self.curves)])
        self.previous = None
        self.observed = None

    def quality(self, matrix, distance, turf):
        """计算可见模型线的像素误差、白线支持率与受支持曲线数量。"""
        failed = {"score": 0.0, "support": 0.0, "coverage": 0.0, "error_px": 100.0, "p90_error_px": 100.0, "curve_count": 0}
        corners = np.array([[0, 0, 1], [self.length, 0, 1], [self.length, self.width, 1], [0, self.width, 1]])
        denominators = corners @ matrix[2]
        if not np.isfinite(matrix).all() or np.min(denominators) <= 0.01 or np.linalg.det(matrix) <= 1e-6:
            return failed
        pixels = transform_points(self.model, matrix)
        height, width = distance.shape
        visible = np.isfinite(pixels).all(axis=1) & (pixels[:, 0] >= 3) & (pixels[:, 0] < width - 3) & (pixels[:, 1] >= 3) & (pixels[:, 1] < height - 3)
        indices = np.flatnonzero(visible)
        if len(indices) < 100:
            return failed
        span = np.ptp(pixels[visible], axis=0)
        if span[0] < width * 0.45 or span[1] < height * 0.35:
            return failed
        sampled = np.rint(pixels[indices]).astype(int)
        errors = distance[sampled[:, 1], sampled[:, 0]]
        on_turf = turf[sampled[:, 1], sampled[:, 0]] > 0
        support = float(np.mean((errors < 3) & on_turf))
        supported_curves = sum(np.count_nonzero((self.groups[indices] == group) & (errors < 3)) >= 8 for group in np.unique(self.groups[indices]))
        score = float(np.mean(np.maximum(0, 1 - errors / 10) * on_turf))
        coverage = 0.0
        if self.observed is not None:
            distances, _ = cKDTree(pixels[visible]).query(self.observed)
            coverage = float(np.mean(distances < 8))
            recall_score = float(np.mean(np.maximum(0, 1 - distances / 15)))
            score = 2 * score * recall_score / max(1e-6, score + recall_score)
        return {"score": score, "support": support, "coverage": coverage, "error_px": float(np.median(errors)), "p90_error_px": float(np.quantile(errors, 0.9)), "curve_count": int(supported_curves)}

    def refine(self, matrix, tree, shape, iterations=8):
        """迭代匹配模型采样与当前白线像素，鲁棒更新单应性。"""
        height, width = shape
        current = matrix.copy()
        for _ in range(iterations):
            pixels = transform_points(self.model, current)
            visible = np.isfinite(pixels).all(axis=1) & (pixels[:, 0] >= 3) & (pixels[:, 0] < width - 3) & (pixels[:, 1] >= 3) & (pixels[:, 1] < height - 3)
            world = self.model[visible]
            if len(world) < 60:
                break
            distances, nearest = tree.query(pixels[visible])
            good = distances < 15
            if np.count_nonzero(good) < 60:
                break
            targets = tree.data[nearest[good]]
            fitted, _ = cv2.findHomography(world[good], targets, cv2.RANSAC, 3.0)
            if fitted is None or not np.isfinite(fitted).all() or abs(np.linalg.det(fitted)) < 1e-9:
                break
            if np.max(np.linalg.norm(transform_points(world[good], fitted) - pixels[visible][good], axis=1)) > 25:
                break
            current = fitted / fitted[2, 2]
        return current

    def align(self, matrix, distance):
        """最小化模型到白线的连续距离，避免 RANSAC 丢弃尚未对齐的整段圆弧。"""
        height, width = distance.shape
        pixels = transform_points(self.model, matrix)
        visible = np.isfinite(pixels).all(axis=1) & (pixels[:, 0] >= 4) & (pixels[:, 0] < width - 4) & (pixels[:, 1] >= 4) & (pixels[:, 1] < height - 4)
        points = self.model[visible] / [self.length, self.width]
        if len(points) < 100:
            return matrix
        normalized = np.diag([1 / width, 1 / width, 1]) @ matrix @ np.diag([self.length, self.width, 1])
        normalized /= normalized[2, 2]
        smooth = gaussian_filter(distance.astype(float), 0.8)

        def residual(parameters):
            """返回白线距离和越界惩罚，保证优化期间观测集合固定。"""
            projected = transform_points(points, np.append(parameters, 1).reshape(3, 3)) * width
            if not np.isfinite(projected).all():
                return np.full(len(points), 1000.0)
            sampled = map_coordinates(smooth, [projected[:, 1], projected[:, 0]], order=1, mode="nearest")
            outside = np.maximum(0, -projected[:, 0]) + np.maximum(0, projected[:, 0] - width + 1) + np.maximum(0, -projected[:, 1]) + np.maximum(0, projected[:, 1] - height + 1)
            return sampled + outside

        result = least_squares(residual, normalized.ravel()[:8], loss="soft_l1", f_scale=3, max_nfev=45, diff_step=1e-4)
        fitted = np.diag([width, width, 1]) @ np.append(result.x, 1).reshape(3, 3) @ np.diag([1 / self.length, 1 / self.width, 1])
        return fitted / fitted[2, 2]

    def process(self, image, frame_index):
        """返回本源帧的映射、白线折线和质量；不充分时 homography 为 None。"""
        frame, mask, turf, scale = extract_white_lines(image, self.work_width)
        lines = detect_long_lines(mask)
        distance = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
        vertical, horizontal = np.nonzero(mask)
        pixels = np.column_stack((horizontal, vertical))
        tiles = (vertical // 32) * ((mask.shape[1] + 31) // 32) + horizontal // 32
        balanced = []
        for tile in np.unique(tiles):
            members = pixels[tiles == tile]
            if len(members) >= 3:
                balanced.extend(members[[len(members) // 4, len(members) * 3 // 4]])
        self.observed = np.asarray(balanced).reshape(-1, 2)
        record = {"frame_index": int(frame_index), "status": "insufficient_geometry", "homography": None, "support": 0.0, "coverage": 0.0, "error_px": None, "p90_error_px": None, "curve_count": 0}
        if len(horizontal) < 150:
            self.previous = None
            return record, mask
        tree = cKDTree(np.column_stack((horizontal, vertical)))
        candidates = []
        if self.previous is not None:
            previous = self.refine(self.previous, tree, mask.shape, iterations=5)
            candidates.append(self.align(previous, distance))
        if not candidates or not self.reliable(self.quality(candidates[0], distance, turf)):
            candidates.extend(circle_homographies(mask, lines, self.length, self.width))
            candidates.extend(rectangle_homographies(mask, lines, self.length, self.width))
        if not candidates:
            self.previous = None
            return record, mask
        scored = sorted(candidates, key=lambda matrix: -self.quality(matrix, distance, turf)["score"])
        finalists = [self.align(matrix, distance) for matrix in scored[:3]]
        fitted = max(finalists, key=lambda matrix: self.quality(matrix, distance, turf)["score"])
        quality = self.quality(fitted, distance, turf)
        record.update({key: value for key, value in quality.items() if key != "score"})
        record["error_px"] /= scale
        record["p90_error_px"] /= scale
        if self.reliable(quality):
            self.previous = fitted
            source_to_pitch = np.linalg.inv(fitted) @ np.diag([scale, scale, 1])
            record.update(status="valid", homography=(source_to_pitch / source_to_pitch[2, 2]).tolist())
        else:
            self.previous = None
        return record, mask

    @staticmethod
    def reliable(quality):
        """要求白线支持、观测覆盖和至少四条曲线同时成立，拒绝局部巧合。"""
        return quality["support"] >= 0.65 and quality["coverage"] >= 0.3 and quality["curve_count"] >= 4 and quality["error_px"] <= 3
