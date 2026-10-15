from dataclasses import asdict, dataclass
from pathlib import Path
import math

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    video: Path
    weights: Path
    output_root: Path
    players: Path
    match_id: str
    clip_id: str
    period: int = 1
    match_offset_s: float | None = None
    start_frame: int = 0
    end_frame: int | None = None
    stride: int = 1
    imgsz: int = 640
    confidence: float = 0.25
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0
    speed_limit_mps: float = 12.0
    min_cutoff_hz: float = 1.0
    filter_beta: float = 0.0

    def snapshot(self):
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}


def load_config(path: str | Path) -> Config:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("配置必须是 YAML 对象")
    return config_from_dict(payload)


def config_from_dict(payload: dict) -> Config:
    values = dict(payload)
    for key, default in [("output_root", "outputs"), ("players", "data/players.yaml")]:
        values.setdefault(key, default)
    for key in ["video", "weights", "output_root", "players"]:
        if not values.get(key):
            raise ValueError(f"缺少配置：{key}")
        candidate = Path(values[key])
        values[key] = candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()
    config = Config(**values)
    for key in ["start_frame", "stride", "imgsz", "period"]:
        value = getattr(config, key)
        if type(value) is not int or value < (0 if key == "start_frame" else 1):
            raise ValueError(f"{key} 必须是有效整数")
    if config.end_frame is not None and (type(config.end_frame) is not int or config.end_frame <= config.start_frame):
        raise ValueError("end_frame 必须大于 start_frame")
    for key in ["confidence", "pitch_length_m", "pitch_width_m", "speed_limit_mps", "min_cutoff_hz"]:
        value = getattr(config, key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} 必须为有限正数")
    if config.confidence >= 1:
        raise ValueError("confidence 必须小于 1")
    if not math.isfinite(config.filter_beta) or config.filter_beta < 0:
        raise ValueError("filter_beta 必须为有限非负数")
    if config.match_offset_s is not None and (not math.isfinite(config.match_offset_s) or config.match_offset_s < 0):
        raise ValueError("match_offset_s 必须为空或有限非负数")
    if not config.match_id or not config.clip_id:
        raise ValueError("match_id 和 clip_id 不能为空")
    return config
