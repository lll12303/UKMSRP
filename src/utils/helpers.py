"""
helpers.py

Small general-purpose helper utilities used across the project.
Keep functions small and dependency-free so they are easy to test.
"""

import os
import json
import yaml
import pandas as pd
from typing import Any, Dict


def ensure_dir(path: str):
    """Ensure a directory exists."""
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def read_csv_safe(path: str, **kwargs) -> pd.DataFrame:
    """
    Read CSV with a safety wrapper.
    Default: low_memory=False, keep default dtype inference.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, low_memory=False, **kwargs)


def write_json(obj: Any, path: str, indent: int = 2):
    """Write object to JSON (ensures directory)."""
    ensure_dir(path)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def read_yaml(path: str) -> Dict:
    """Read YAML file to dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_dataframe(df, path: str, index: bool = False):
    """Save DataFrame to CSV ensuring directory exists."""
    ensure_dir(path)
    df.to_csv(path, index=index)


def standardize_column_names(df: pd.DataFrame, lower: bool = True, replace_spaces: bool = True) -> pd.DataFrame:
    """Basic column name standardization: lower-case, strip spaces, replace spaces with underscore."""
    cols = list(df.columns)
    new_cols = []
    for c in cols:
        nc = c
        if lower:
            nc = nc.lower()
        if replace_spaces:
            nc = nc.replace(" ", "_")
        nc = nc.strip()
        new_cols.append(nc)
    df = df.copy()
    df.columns = new_cols
    return df
