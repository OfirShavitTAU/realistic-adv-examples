"""Portable JSON and atomic artifact writes for experiment output."""

import csv
import dataclasses
import json
import math
import os
from collections.abc import Mapping
from enum import Enum
from pathlib import Path

import numpy as np
import torch


"""
    Converts original counters and metadata into JSON-compatible values.
    Nonfinite numbers become null; unsupported objects raise instead of being silently stringified.
"""
def json_value(value: object) -> object:
    if isinstance(value, Enum):
        return json_value(value.value)

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.Tensor):
        return json_value(value.detach().cpu().tolist())

    if isinstance(value, (np.ndarray, np.generic)):
        return json_value(value.tolist())

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: json_value(getattr(value, field.name)) for field in dataclasses.fields(value)}

    if isinstance(value, Mapping):
        return {str(key.value if isinstance(key, Enum) else key): json_value(item) for key, item in value.items()}

    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]

    raise TypeError("Unsupported output metadata type: " + type(value).__name__)


"""
    Atomically replaces a JSON file after flushing its complete new contents.
    Existing artifacts remain readable if writing the replacement fails.
"""
def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)

    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(json_value(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())

    temporary.replace(path)


"""
    Atomically rewrites the summary so readers see a complete snapshot after each task.
"""
def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")

    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())

    temporary.replace(path)


"""
    Saves exact tensor values separately from rounded display images.
"""
def write_tensor(path: Path, tensor: torch.Tensor) -> None:
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)

    with temporary.open("wb") as stream:
        torch.save(tensor.detach().cpu(), stream)
        stream.flush()
        os.fsync(stream.fileno())

    temporary.replace(path)
