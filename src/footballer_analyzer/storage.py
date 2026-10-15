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
    return f"{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"


def file_hash(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


def read_yaml(path: Path, default=None):
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else default


def read_rows(path: Path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
        yield from batch.to_pylist()


def versions():
    names = ["footballer-analyzer", "ultralytics", "supervision", "opencv-python", "numpy", "pyarrow", "scikit-learn", "PyYAML"]
    return {name: metadata.version(name) for name in names}


class RowWriter:
    def __init__(self, path: Path, schema, batch_size=2048):
        self.path = path
        self.temporary = path.with_suffix(".parquet.partial")
        self.writer = pq.ParquetWriter(self.temporary, schema, compression="zstd")
        self.schema = schema
        self.batch_size = batch_size
        self.buffer = []

    def append(self, row):
        self.buffer.append(row)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        if self.buffer:
            self.writer.write_table(pa.Table.from_pylist(self.buffer, schema=self.schema))
            self.buffer.clear()

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        try:
            if error_type is None:
                self.flush()
        finally:
            self.writer.close()
        if error_type is None:
            os.replace(self.temporary, self.path)
