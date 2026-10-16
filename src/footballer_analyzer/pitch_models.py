"""定义十一人制和八人制场地参数及地面参考点；坐标单位为米。"""

import math


EIGHT_SOURCE = "https://www.jfa.jp/youth_development/players_first/pdf/8system.pdf"
PRESETS = {
    "eleven": {"label": "十一人制", "length_m": 105.0, "width_m": 68.0,
               "goal_width_m": 7.32, "goal_height_m": 2.44, "penalty_depth_m": 16.5,
               "goal_area_depth_m": 5.5, "penalty_mark_m": 11.0, "circle_radius_m": 9.15,
               "corner_radius_m": 1.0, "source": "IFAB 足球竞赛规则"},
    "eight": {"label": "八人制", "length_m": 68.0, "width_m": 50.0,
              "goal_width_m": 5.0, "goal_height_m": 2.15, "penalty_depth_m": 12.0,
              "goal_area_depth_m": 4.0, "penalty_mark_m": 8.0, "circle_radius_m": 7.0,
              "corner_radius_m": 1.0, "source": EIGHT_SOURCE},
}


def pitch_spec(model="eleven", length=None, width=None):
    """返回独立参数快照；允许调整整体长宽，拒绝无法容纳标线的尺寸。"""
    if model not in PRESETS:
        raise ValueError("场地模型仅支持 eleven 或 eight")
    spec = {"model": model, "template_version": 1, **PRESETS[model]}
    spec["length_m"] = float(spec["length_m"] if length is None else length)
    spec["width_m"] = float(spec["width_m"] if width is None else width)
    if not all(math.isfinite(spec[key]) and spec[key] > 0 for key in ["length_m", "width_m"]):
        raise ValueError("球场尺寸必须为有限正数")
    spec["penalty_width_m"] = spec["goal_width_m"] + 2 * spec["penalty_depth_m"]
    spec["goal_area_width_m"] = spec["goal_width_m"] + 2 * spec["goal_area_depth_m"]
    if spec["length_m"] <= 2 * spec["penalty_depth_m"] or spec["width_m"] <= spec["penalty_width_m"]:
        raise ValueError("球场尺寸不足以容纳所选模型的禁区")
    return spec


def pitch_landmarks(spec):
    """从参数快照生成具名地面参考点，不包含横梁等离地点。"""
    length, width = spec["length_m"], spec["width_m"]
    landmarks = []

    def add(identifier, label, horizontal, vertical):
        """追加一个稳定标识的中文参考点。"""
        landmarks.append({"id": identifier, "label": label, "pitch": [horizontal, vertical]})

    for side, origin, sign, name in [("left", 0, 1, "左侧"), ("right", length, -1, "右侧")]:
        for edge, vertical, edge_name in [("far", 0, "上"), ("near", width, "下")]:
            add(f"{side}_corner_{edge}", f"{name}{edge_name}场角", origin, vertical)
        for area, depth_key, width_key, area_name in [
            ("penalty", "penalty_depth_m", "penalty_width_m", "罚球区"),
            ("goal_area", "goal_area_depth_m", "goal_area_width_m", "球门区"),
        ]:
            for edge, vertical, edge_name in [("far", (width - spec[width_key]) / 2, "上"), ("near", (width + spec[width_key]) / 2, "下")]:
                add(f"{side}_{area}_{edge}_back", f"{name}{area_name}{edge_name}底角", origin, vertical)
                add(f"{side}_{area}_{edge}_front", f"{name}{area_name}{edge_name}前角", origin + sign * spec[depth_key], vertical)
        add(f"{side}_penalty_mark", f"{name}罚球点", origin + sign * spec["penalty_mark_m"], width / 2)
        for edge, vertical, edge_name in [("far", (width - spec["goal_width_m"]) / 2, "上"), ("near", (width + spec["goal_width_m"]) / 2, "下")]:
            add(f"{side}_goal_{edge}", f"{name}{edge_name}门柱落地点", origin, vertical)
    add("center", "开球点", length / 2, width / 2)
    for edge, vertical, edge_name in [("far", 0, "上"), ("near", width, "下")]:
        add(f"halfway_{edge}", f"中线{edge_name}边线交点", length / 2, vertical)
    radius = spec["circle_radius_m"]
    for direction, vertical, label in [("far", width / 2 - radius, "中圈与中线上交点"), ("near", width / 2 + radius, "中圈与中线下交点")]:
        add(f"center_circle_{direction}", label, length / 2, vertical)
    return landmarks
