from __future__ import annotations

import gzip
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def jsonl_open(path: str | Path, mode: str):
    """open() that transparently switches to gzip when the path ends in .gz."""
    if str(path).endswith('.gz'):
        kwargs = {'compresslevel': 4} if 'w' in mode else {}
        return gzip.open(path, mode + 't', encoding='utf-8', **kwargs)
    return open(path, mode, encoding='utf-8')


def resolve_cases_path(cases_dir: str | Path, split: str) -> Path | None:
    """Resolve cases_{split}.jsonl or cases_{split}.jsonl.gz; None if neither exists."""
    cases_dir = Path(cases_dir)
    plain = cases_dir / f'cases_{split}.jsonl'
    if plain.exists():
        return plain
    gz = cases_dir / f'cases_{split}.jsonl.gz'
    if gz.exists():
        return gz
    return None


def count_jsonl_lines(path: str | Path) -> int:
    """Count non-blank lines without materializing rows."""
    n = 0
    with jsonl_open(path, 'r') as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def read_jsonl(path: str | Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with jsonl_open(path, 'r') as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with jsonl_open(path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    if path.suffix in {'.jsonl', '.json'}:
        if path.suffix == '.jsonl':
            return pd.DataFrame(read_jsonl(path))
        return pd.read_json(path)
    if path.suffix == '.csv':
        return pd.read_csv(path)
    raise ValueError(f'Unsupported table format: {path}')


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if path.suffix == '.parquet':
        df.to_parquet(path, index=False)
    elif path.suffix == '.csv':
        df.to_csv(path, index=False)
    elif path.suffix == '.jsonl':
        write_jsonl(df.to_dict('records'), path)
    else:
        raise ValueError(f'Unsupported table format: {path}')


def coalesce(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ''):
            return row[key]
    return default


def normalize_timestamp(ts: Any) -> int:
    """Normalize an int/float/ISO-string timestamp to a stable sortable integer."""
    if pd.isna(ts):
        return 0
    if isinstance(ts, (int, np.integer)):
        return int(ts)
    if isinstance(ts, float):
        return int(ts)
    s = str(ts)
    if s.isdigit():
        return int(s)
    parsed = pd.to_datetime(s, errors='coerce')
    if pd.isna(parsed):
        return 0
    return int(parsed.timestamp() * 1000)


def safe_mean(values: list[float], default: float = 0.0) -> float:
    return float(np.mean(values)) if values else default


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    b_norm = np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    return (a / a_norm) @ (b / b_norm).T


def cosine_vec(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / den)
