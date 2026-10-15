"""保存分析产物：原子替换 JSON/YAML，分批写入 Parquet，并记录摘要与版本。"""

from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import hashlib
import json
import os
import uuid

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

TRACK_SCHEMA = pa.schema([
    ("frame_index", pa.int64()), ("timestamp_s", pa.float64()), ("track_id", pa.int64()),
    ("bbox_x1", pa.float64()), ("bbox_y1", pa.float64()),
    ("bbox_x2", pa.float64()), ("bbox_y2", pa.float64()),
    ("confidence", pa.float64()), ("class_id", pa.int64()), ("team_id", pa.string()),
])


def new_id(prefix):
    """返回由前缀、UTC 时间和随机后缀组成的产物标识。"""
    return f"{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"


def file_hash(path: Path):
    """分块读取文件，返回 SHA256 十六进制摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload):
    """以 UTF-8 原子写入 JSON；拒绝 NaN/Infinity，并自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path):
    """读取 UTF-8 JSON；文件或格式异常向调用方传播。"""
    return json.loads(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, payload):
    """以 UTF-8 原子写入安全 YAML，保留中文与字段顺序。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


def read_yaml(path: Path, default=None):
    """安全解析 YAML；文件不存在时返回 default，空文件解析为 None。"""
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else default


def read_rows(path: Path):
    """分批读取 Parquet，逐行产出字典，避免一次载入整张表。"""
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
        yield from batch.to_pylist()


def versions():
    """返回运行清单需要记录的应用与核心依赖版本。"""
    names = ["footballer-analyzer", "ultralytics", "supervision", "opencv-python", "numpy", "pyarrow", "scikit-learn", "PyYAML"]
    return {name: metadata.version(name) for name in names}


class RowWriter:
    """按固定 schema 缓冲写入 Parquet，仅在正常退出时发布正式文件。"""

    def __init__(self, path: Path, schema, batch_size=2048):
        """打开 .parquet.partial 临时文件，并设置行结构和批次大小。"""
        self.path = path
        self.temporary = path.with_suffix(".parquet.partial")
        self.writer = pq.ParquetWriter(self.temporary, schema, compression="zstd")
        self.schema = schema
        self.batch_size = batch_size
        self.buffer = []

    def append(self, row):
        """追加一行字典，缓冲达到批次大小时写盘。"""
        self.buffer.append(row)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        """按 schema 写出当前缓冲并清空；无数据时不操作。"""
        if self.buffer:
            self.writer.write_table(pa.Table.from_pylist(self.buffer, schema=self.schema))
            self.buffer.clear()

    def __enter__(self):
        """返回自身，供 with 语句管理写入生命周期。"""
        return self

    def __exit__(self, error_type, error, traceback):
        """关闭写入器；成功时原子发布文件，异常时保留临时文件并继续抛出。"""
        try:
            if error_type is None:
                self.flush()
        finally:
            self.writer.close()
        if error_type is None:
            os.replace(self.temporary, self.path)
