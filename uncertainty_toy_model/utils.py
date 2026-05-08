"""Small utilities for reproducible experiments and readable output."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_float(value: float) -> str:
    return f"{value:.2f}"


def print_table(rows: list[dict[str, object]], columns: list[str]) -> None:
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows))
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    separator = "-+-".join("-" * widths[column] for column in columns)
    print(header)
    print(separator)
    for row in rows:
        print(" | ".join(str(row[column]).ljust(widths[column]) for column in columns))
