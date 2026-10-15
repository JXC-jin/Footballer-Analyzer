import math

import numpy as np
import pyarrow as pa

from .calibration import project_observation
from .identity import resolve_identity

MOTION_SCHEMA = pa.schema([
    ("frame_index", pa.int64()), ("timestamp_s", pa.float64()), ("track_id", pa.int64()),
    ("player_id", pa.string()), ("team_id", pa.string()), ("segment_id", pa.int64()),
    ("x_m", pa.float64()), ("y_m", pa.float64()), ("x_smooth_m", pa.float64()), ("y_smooth_m", pa.float64()),
    ("valid", pa.bool_()), ("reason", pa.string()), ("delta_t_s", pa.float64()),
    ("distance_delta_m", pa.float64()), ("speed_mps", pa.float64()),
])

SUMMARY_SCHEMA = pa.schema([
    ("match_id", pa.string()), ("clip_id", pa.string()), ("player_id", pa.string()), ("team_id", pa.string()),
    ("distance_m", pa.float64()), ("valid_seconds", pa.float64()), ("coverage", pa.float64()),
    ("avg_speed_mps", pa.float64()),
])


class OneEuro:
    def __init__(self, position, min_cutoff, beta):
        self.position = np.asarray(position, dtype=float)
        self.previous_raw = self.position.copy()
        self.derivative = np.zeros(2)
        self.min_cutoff = min_cutoff
        self.beta = beta

    @staticmethod
    def alpha(cutoff, delta_t):
        return 1 / (1 + 1 / (2 * math.pi * cutoff * delta_t))

    def update(self, position, delta_t):
        position = np.asarray(position)
        derivative_alpha = self.alpha(1.0, delta_t)
        self.derivative += derivative_alpha * ((position - self.previous_raw) / delta_t - self.derivative)
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.derivative)
        self.position += self.alpha(cutoff, delta_t) * (position - self.position)
        self.previous_raw = position.copy()
        return self.position.copy()


def calculate_motion(rows, manifest, calibration, bindings):
    config = manifest["config"]
    states = {}
    segment_id = 0
    for row in rows:
        player_id, team_id, binding_index, identity_reason = resolve_identity(row, bindings)
        world_x, world_y, projection_reason = project_observation(row, calibration)
        reason = projection_reason or identity_reason
        result = {
            "frame_index": row["frame_index"], "timestamp_s": row["timestamp_s"], "track_id": row["track_id"],
            "player_id": player_id, "team_id": team_id, "segment_id": None,
            "x_m": world_x, "y_m": world_y, "x_smooth_m": None, "y_smooth_m": None,
            "valid": reason is None, "reason": reason,
            "delta_t_s": None, "distance_delta_m": None, "speed_mps": None,
        }
        previous = states.get(row["track_id"])
        if reason:
            states.pop(row["track_id"], None)
            yield result
            continue
        position = np.array([world_x, world_y])
        continuous = previous is not None and previous["binding_index"] == binding_index and row["frame_index"] - previous["frame_index"] == config["stride"]
        if continuous:
            delta_t = row["timestamp_s"] - previous["timestamp_s"]
            raw_speed = float(np.linalg.norm(position - previous["raw"])) / delta_t
            if raw_speed > config["speed_limit_mps"]:
                result.update(valid=False, reason="speed_jump")
                states.pop(row["track_id"], None)
                yield result
                continue
            smooth = previous["filter"].update(position, delta_t)
            distance = float(np.linalg.norm(smooth - previous["smooth"]))
            result.update(segment_id=previous["segment_id"], delta_t_s=delta_t, distance_delta_m=distance, speed_mps=distance / delta_t)
            current_filter = previous["filter"]
        else:
            segment_id += 1
            result["segment_id"] = segment_id
            current_filter = OneEuro(position, config["min_cutoff_hz"], config["filter_beta"])
            smooth = position.copy()
        result.update(x_smooth_m=float(smooth[0]), y_smooth_m=float(smooth[1]))
        states[row["track_id"]] = {
            "frame_index": row["frame_index"], "timestamp_s": row["timestamp_s"], "binding_index": binding_index,
            "raw": position, "smooth": smooth, "filter": current_filter, "segment_id": result["segment_id"],
        }
        yield result


def summarize(rows, manifest, bindings):
    players = {entry["player_id"]: entry["team_id"] for entry in bindings if entry["action"] == "bind"}
    totals = {player_id: [0.0, 0.0] for player_id in players}
    for row in rows:
        if row["valid"] and row["delta_t_s"] is not None:
            totals[row["player_id"]][0] += row["distance_delta_m"]
            totals[row["player_id"]][1] += row["delta_t_s"]
    duration = (manifest["end_frame"] - manifest["start_frame"]) / manifest["video"]["fps"]
    return [{
        "match_id": manifest["config"]["match_id"], "clip_id": manifest["config"]["clip_id"],
        "player_id": player_id, "team_id": players[player_id],
        "distance_m": distance if seconds else None, "valid_seconds": seconds, "coverage": seconds / duration,
        "avg_speed_mps": distance / seconds if seconds else None,
    } for player_id, (distance, seconds) in totals.items()]
