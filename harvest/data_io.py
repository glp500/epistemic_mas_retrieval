"""Shared JSON, JSONL and CSV serialization for experiment artifacts."""

import csv
import datetime
import hashlib
import json

import numpy as np


def plain(value):
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, datetime.date):  # YAML reads unquoted dates as date objects
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(
        json.dumps(plain(value), sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def file_digest(path):
    """Hash large artifacts without loading a second copy into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(plain(value), indent=2, ensure_ascii=False) + "\n")


def append(stream, value):
    stream.write(json.dumps(plain(value), ensure_ascii=False) + "\n")
    stream.flush()


def iter_rows(path):
    """Read one JSONL record at a time, including large gameplay logs."""
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def read_rows(path):
    return list(iter_rows(path))


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
