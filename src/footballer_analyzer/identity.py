"""维护跨场球员档案与本次运行的区间绑定；所有源帧区间均为左闭右开。"""

from .storage import read_yaml


def load_players(path):
    """读取球员档案并按 player_id 索引；缺文件返回空字典，格式或重复 ID 报错。"""
    payload = read_yaml(path, {"players": []})
    if not isinstance(payload, dict) or not isinstance(payload.get("players"), list):
        raise ValueError("球员档案格式必须为 players 列表")
    players = {}
    for entry in payload["players"]:
        player_id = entry.get("player_id")
        if not isinstance(player_id, str) or not player_id.strip() or player_id in players or not str(entry.get("display_name", "")).strip():
            raise ValueError("球员 ID 必须非空、唯一，且需提供姓名")
        players[player_id] = entry
    return players


def validate_bindings(payload, manifest, players, catalog):
    """校验并返回绑定列表；错误运行、未知身份、越界、重叠或队伍冲突时报错。

    同一球员可绑定多个不重叠轨迹区间；action 为 bind 或 exclude。
    """
    if not isinstance(payload, dict) or payload.get("run_id") != manifest["run_id"] or not isinstance(payload.get("bindings"), list):
        raise ValueError("身份绑定的 run_id 或列表格式不匹配")
    entries = payload["bindings"]
    track_ids = {entry["track_id"] for entry in catalog}
    for entry in entries:
        if entry.get("track_id") not in track_ids:
            raise ValueError("绑定引用了不存在的轨迹")
        start, end = entry.get("start_frame"), entry.get("end_frame")
        if type(start) is not int or type(end) is not int or not manifest["start_frame"] <= start < end <= manifest["end_frame"]:
            raise ValueError("绑定帧区间无效或越界")
        if entry.get("action") not in ["bind", "exclude"]:
            raise ValueError("绑定 action 必须为 bind 或 exclude")
        if entry["action"] == "bind":
            if entry.get("player_id") not in players or entry.get("team_id") not in ["home", "away", "unknown"]:
                raise ValueError("绑定球员不存在或球队不是 home/away/unknown")
    for index, first in enumerate(entries):
        for second in entries[index + 1:]:
            overlap = max(first["start_frame"], second["start_frame"]) < min(first["end_frame"], second["end_frame"])
            if overlap and first["track_id"] == second["track_id"]:
                raise ValueError("同一轨迹的绑定或排除区间重叠")
            if first["action"] == second["action"] == "bind" and first["player_id"] == second["player_id"]:
                if overlap:
                    raise ValueError("同一球员同时绑定了两条轨迹")
                if first["team_id"] != second["team_id"]:
                    raise ValueError("同一球员的本场球队不一致")
    return entries


def resolve_identity(row, bindings):
    """按轨迹和源帧查找绑定，返回球员 ID、球队、绑定序号及排除/未绑定原因。"""
    for binding_index, entry in enumerate(bindings):
        if entry["track_id"] == row["track_id"] and entry["start_frame"] <= row["frame_index"] < entry["end_frame"]:
            if entry["action"] == "exclude":
                return None, "unknown", binding_index, "excluded"
            return entry["player_id"], entry["team_id"], binding_index, None
    return None, row.get("team_id", "unknown"), None, "unbound"
