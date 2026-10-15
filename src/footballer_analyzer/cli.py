import argparse
import json
from pathlib import Path
import sys

from .config import ROOT, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="FootballerAnalyzer · 足球轨迹与球场位置复盘")
    commands = parser.add_subparsers(dest="command", required=True)
    track_command = commands.add_parser("track", help="检测跟踪并保存原始轨迹")
    track_command.add_argument("--config", required=True)
    calibration_command = commands.add_parser("calibrate", help="从 YAML 对应点保存标定，也可在前端点选")
    calibration_command.add_argument("--run", required=True, type=Path)
    calibration_command.add_argument("--points", required=True, type=Path)
    report_command = commands.add_parser("report", help="复用缓存轨迹生成个人跑动报告")
    report_command.add_argument("--run", required=True, type=Path)
    history_command = commands.add_parser("history", help="查询球员在选定报告中的记录")
    history_command.add_argument("--player", required=True)
    history_command.add_argument("--reports", nargs="+", required=True, type=Path)
    serve_command = commands.add_parser("serve", help="启动本地球员追踪与球场位置调试台")
    serve_command.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    serve_command.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    try:
        if args.command == "track":
            from .pipeline import track

            track(load_config(args.config))
        elif args.command == "calibrate":
            from .calibration import save_calibration
            from .storage import read_yaml

            save_calibration(args.run.resolve(), read_yaml(args.points))
        elif args.command == "report":
            from .reporting import create_report

            create_report(args.run.resolve())
        elif args.command == "history":
            from .reporting import history

            print(json.dumps(history(args.player, args.reports), ensure_ascii=False, indent=2))
        elif args.command == "serve":
            from .server import serve

            serve(args.output_root, args.port)
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1
