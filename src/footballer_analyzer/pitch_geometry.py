"""提取草地区域内的白线，并提供标准足球场的米制线段与圆弧模型。"""

from itertools import combinations

import cv2
import numpy as np


def pitch_polylines(length=105.0, width=68.0, spacing=0.7):
    """返回具名场地折线，坐标单位米；禁区、球门区与圆半径采用标准尺寸。"""
    curves = []

    def line(name, start, end):
        """按米制步长采样直线并加入模型。"""
        count = max(2, int(np.linalg.norm(np.subtract(end, start)) / spacing) + 1)
        curves.append((name, np.linspace(start, end, count)))

    def arc(name, center, radius, start, end):
        """按弧长采样圆弧，角度使用弧度。"""
        angles = np.linspace(start, end, max(16, int((end - start) * radius / spacing)))
        curves.append((name, np.column_stack((np.cos(angles), np.sin(angles))) * radius + center))

    line("touch_far", (0, 0), (length, 0))
    line("touch_near", (0, width), (length, width))
    line("goal_left", (0, 0), (0, width))
    line("goal_right", (length, 0), (length, width))
    line("halfway", (length / 2, 0), (length / 2, width))
    arc("center_circle", (length / 2, width / 2), 9.15, 0, 2 * np.pi)
    for side, origin, sign in [("left", 0, 1), ("right", length, -1)]:
        for label, depth, half_width in [("penalty", 16.5, 20.16), ("goal_area", 5.5, 9.16)]:
            front = origin + sign * depth
            low, high = width / 2 - half_width, width / 2 + half_width
            line(f"{side}_{label}_far", (origin, low), (front, low))
            line(f"{side}_{label}_front", (front, low), (front, high))
            line(f"{side}_{label}_near", (origin, high), (front, high))
        angle = np.arccos(5.5 / 9.15)
        start = -angle if sign == 1 else np.pi - angle
        end = angle if sign == 1 else np.pi + angle
        arc(f"{side}_penalty_arc", (origin + sign * 11, width / 2), 9.15, start, end)
    return curves


def transform_points(points, matrix):
    """将二维点经单应性变换，返回二维坐标；无穷远点由调用方过滤。"""
    homogeneous = np.column_stack((np.asarray(points), np.ones(len(points)))) @ np.asarray(matrix).T
    divisor = homogeneous[:, 2:3]
    return homogeneous[:, :2] / np.where(np.abs(divisor) > 1e-9, divisor, np.nan)


def extract_white_lines(image, work_width=960):
    """返回缩放图、白线掩膜、草地掩膜与缩放比；抑制看台、广告和大块白衣。"""
    scale = min(1.0, work_width / image.shape[1])
    frame = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, np.array([25, 45, 35]), np.array([95, 255, 255]))
    neighborhood = cv2.boxFilter((grass > 0).astype(np.float32), -1, (25, 25))
    grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(grass, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    turf = np.zeros_like(grass)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > turf.size * 0.12:
            cv2.drawContours(turf, [cv2.convexHull(contour)], -1, 255, -1)
            turf = cv2.dilate(turf, np.ones((9, 9), np.uint8))
    value = hsv[:, :, 2]
    contrast = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, np.ones((13, 13), np.uint8))
    saturation_contrast = cv2.morphologyEx(hsv[:, :, 1], cv2.MORPH_BLACKHAT, np.ones((13, 13), np.uint8))
    white = ((hsv[:, :, 1] < 150) & (value > 110) & ((contrast > 16) | (saturation_contrast > 18)) & (turf > 0) & (neighborhood > 0.45)).astype(np.uint8) * 255
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(white)
    for component in range(1, component_count):
        if stats[component, cv2.CC_STAT_AREA] < 5:
            white[labels == component] = 0
    return frame, white, turf, scale


def detect_long_lines(mask):
    """合并 Hough 检出的共线线段，返回归一化齐次直线、端点与长度。"""
    segments = cv2.HoughLinesP(mask, 1, np.pi / 720, 35, minLineLength=65, maxLineGap=20)
    groups = []
    if segments is None:
        return groups
    segments = sorted(segments.reshape(-1, 4).astype(float), key=lambda segment: -np.linalg.norm(segment[2:] - segment[:2]))
    for segment in segments:
        start, end = segment[:2], segment[2:]
        equation = np.cross([*start, 1.0], [*end, 1.0])
        equation /= np.linalg.norm(equation[:2])
        if any(max(abs(np.dot(group["equation"], [*start, 1])), abs(np.dot(group["equation"], [*end, 1]))) < 5 for group in groups):
            continue
        groups.append({"equation": equation, "start": start, "end": end, "length": float(np.linalg.norm(end - start))})
    return groups[:16]


def circle_candidates(mask, lines):
    """移除长直线后拟合中圈候选，返回椭圆；残缺或尺寸不合理的轮廓被拒绝。"""
    curved = mask.copy()
    height, width = mask.shape
    for line in lines:
        if line["length"] < width * 0.15:
            continue
        direction = line["end"] - line["start"]
        direction /= np.linalg.norm(direction)
        cv2.line(curved, np.rint(line["start"] - direction * width * 2).astype(int), np.rint(line["end"] + direction * width * 2).astype(int), 0, 5)
    contours, _ = cv2.findContours(curved, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for contour in contours:
        if len(contour) < 70:
            continue
        ellipse = cv2.fitEllipse(contour)
        center, axes, angle = ellipse
        small, large = sorted(axes)
        if not 20 < small < height * 0.4 or not 100 < large < width * 0.75 or not 1.4 < large / small < 9:
            continue
        if not 0 < center[0] < width or not height * 0.15 < center[1] < height * 0.85:
            continue
        radians = np.deg2rad(angle)
        rotation = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
        local = (contour.reshape(-1, 2) - center) @ rotation
        error = np.abs(np.sum((local / (np.array(axes) / 2)) ** 2, axis=1) - 1)
        if np.median(error) < 0.3 and np.quantile(error, 0.9) < 0.8:
            if not any(np.linalg.norm(np.subtract(center, previous[0])) < 15 for previous in candidates):
                candidates.append(ellipse)
    for line in lines:
        equation = line["equation"]
        if abs(equation[0]) > 0.9:
            center_y = height * 0.43
            center_x = -(equation[1] * center_y + equation[2]) / equation[0]
            candidates.append(((center_x, center_y), (height * 0.16, width * 0.4), 90))
    refined = []
    random = np.random.default_rng(42)
    vertical, horizontal = np.nonzero(curved)
    pixels = np.column_stack((horizontal, vertical)).astype(np.float32)
    for center, axes, angle in candidates:
        nearby = pixels[(np.abs(pixels[:, 0] - center[0]) < max(axes) * 0.9) & (np.abs(pixels[:, 1] - center[1]) < min(axes) * 1.4)]
        if len(nearby) < 80:
            continue
        best, best_inliers, best_score = None, None, 0
        for _ in range(350):
            ellipse = cv2.fitEllipse(nearby[random.choice(len(nearby), 6, replace=False)])
            ellipse_center, ellipse_axes, ellipse_angle = ellipse
            if not 20 < min(ellipse_axes) < height * 0.4 or not 100 < max(ellipse_axes) < width * 0.75 or not 1.4 < max(ellipse_axes) / min(ellipse_axes) < 9:
                continue
            if abs(ellipse_angle - 90) > 25:
                continue
            radians = np.deg2rad(ellipse_angle)
            rotation = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
            local = (nearby - ellipse_center) @ rotation
            radius = np.array(ellipse_axes) / 2
            implicit = np.sum((local / radius) ** 2, axis=1) - 1
            gradient = 2 * np.linalg.norm(local / radius ** 2, axis=1)
            inliers = np.abs(implicit) / np.maximum(gradient, 1e-5) < 1.5
            angles = np.arctan2(local[inliers, 1] / radius[1], local[inliers, 0] / radius[0])
            coverage = len(np.unique(np.floor((angles + np.pi) / (np.pi / 8))))
            score = np.count_nonzero(inliers) if coverage >= 10 else 0
            if score > best_score:
                best, best_inliers, best_score = ellipse, inliers, score
        if best is not None:
            refined.append(cv2.fitEllipse(nearby[best_inliers]))
    return refined


def intersect_lines(first, second):
    """求齐次直线交点，平行时返回 None。"""
    point = np.cross(first, second)
    return point[:2] / point[2] if abs(point[2]) > 1e-8 else None


def circle_homographies(mask, lines, length, width):
    """由中圈、中线和两条边线自动求候选映射；图像向右约定为球场 x 正向。"""
    image_height, image_width = mask.shape
    horizontal = [line for line in lines if abs(line["equation"][0]) < 0.45]
    vertical = [line for line in lines if abs(line["equation"][0]) > 0.85]
    matrices = []
    for center, axes, angle in circle_candidates(mask, lines):
        radians = np.deg2rad(angle)
        rotation = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
        quadratic = rotation @ np.diag(1 / (np.array(axes) / 2) ** 2) @ rotation.T
        center = np.asarray(center)
        conic = np.block([[quadratic, (-quadratic @ center)[:, None]], [(-center @ quadratic)[None, :], np.array([[center @ quadratic @ center - 1]])]])
        for middle in vertical:
            equation = middle["equation"]
            if abs(np.dot(equation, [*center, 1])) > min(axes) * 0.3:
                continue
            origin = center - equation[:2] * np.dot(equation, [*center, 1])
            direction = np.array([-equation[1], equation[0]])
            offset = origin - center
            roots = np.roots([direction @ quadratic @ direction, 2 * offset @ quadratic @ direction, offset @ quadratic @ offset - 1])
            if not np.isreal(roots).all():
                continue
            circle_points = sorted([origin + float(root) * direction for root in roots], key=lambda point: point[1])
            for far in horizontal:
                far_point = intersect_lines(equation, far["equation"])
                if far_point is None or not 0 <= far_point[1] < circle_points[0][1] - 15:
                    continue
                for near in horizontal:
                    near_point = intersect_lines(equation, near["equation"])
                    if near_point is None or not circle_points[1][1] + 30 < near_point[1] < image_height * 1.2:
                        continue
                    observed = np.array([far_point, *circle_points, near_point])
                    targets = np.array([-width / 2, -9.15, 9.15, width / 2])
                    design, values = [], []
                    for coordinate, (horizontal_pixel, vertical_pixel) in zip(targets, observed):
                        design.extend([[coordinate, 0, 1, 0, -horizontal_pixel * coordinate], [0, coordinate, 0, 1, -vertical_pixel * coordinate]])
                        values.extend([horizontal_pixel, vertical_pixel])
                    fitted = np.linalg.lstsq(np.asarray(design), values, rcond=None)[0]
                    second_column = np.array([fitted[0], fitted[1], fitted[4]])
                    third_column = np.array([fitted[2], fitted[3], 1])
                    reprojection = (second_column[None, :] * targets[:, None] + third_column)
                    if np.max(np.linalg.norm(reprojection[:, :2] / reprojection[:, 2:] - observed, axis=1)) > 5:
                        continue
                    vanishing = np.cross(far["equation"], near["equation"])
                    vanishing /= np.linalg.norm(vanishing)
                    radial = 9.15 * vanishing
                    factors = np.roots([radial @ conic @ radial, 2 * third_column @ conic @ radial, third_column @ conic @ third_column])
                    for factor in factors:
                        if not np.isreal(factor):
                            continue
                        first_column = float(np.real(factor)) * vanishing
                        local = np.column_stack((first_column, second_column, third_column))
                        if transform_points([[1, 0]], local)[0, 0] <= third_column[0]:
                            continue
                        offset_matrix = np.array([[1, 0, -length / 2], [0, 1, -width / 2], [0, 0, 1]])
                        matrix = local @ offset_matrix
                        matrices.append(matrix / matrix[2, 2])
    return matrices


def rectangle_homographies(mask, lines, length, width):
    """枚举边线与禁区矩形的几何对应，供中圈不可见时自动初始化。"""
    height, image_width = mask.shape
    horizontal = [line for line in lines if abs(line["equation"][0]) < 0.5]
    transverse = [line for line in lines if abs(line["equation"][0]) > 0.3]
    rectangles = [(0, length, 0, width), (0, length / 2, 0, width), (length / 2, length, 0, width), (16.5, length / 2, 0, width), (length / 2, length - 16.5, 0, width)]
    for origin, sign in [(0, 1), (length, -1)]:
        for depth, half_width in [(16.5, 20.16), (5.5, 9.16), (16.5, width / 2)]:
            first, second = sorted([origin, origin + sign * depth])
            rectangles.append((first, second, width / 2 - half_width, width / 2 + half_width))
    matrices = []
    for far, near in combinations(horizontal, 2):
        if abs(np.dot(far["equation"][:2], near["equation"][:2])) < 0.85:
            continue
        for first, second in combinations(transverse, 2):
            if abs(np.dot(first["equation"][:2], second["equation"][:2])) < 0.7:
                continue
            if abs(np.dot(far["equation"][:2], first["equation"][:2])) > 0.95:
                continue
            corners = [intersect_lines(boundary["equation"], cross["equation"]) for boundary, cross in [(far, first), (far, second), (near, second), (near, first)]]
            if any(corner is None for corner in corners):
                continue
            corners = np.asarray(corners, np.float32)
            if not np.isfinite(corners).all() or not cv2.isContourConvex(corners) or cv2.contourArea(corners) < 5000:
                continue
            if np.any(corners < [-image_width * 0.2, -height * 0.1]) or np.any(corners > [image_width * 1.2, height * 1.2]):
                continue
            if corners[:2, 1].mean() > corners[2:, 1].mean():
                corners = corners[[3, 2, 1, 0]]
            if corners[0, 0] > corners[1, 0]:
                corners = corners[[1, 0, 3, 2]]
            for left, right, top, bottom in rectangles:
                world = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], np.float32)
                matrix = cv2.getPerspectiveTransform(world, corners)
                denominators = np.column_stack(([[0, 0], [length, 0], [length, width], [0, width]], np.ones(4))) @ matrix[2]
                if np.min(denominators) > 0.05 and np.linalg.det(matrix) > 1e-6:
                    matrices.append(matrix)
    return matrices
