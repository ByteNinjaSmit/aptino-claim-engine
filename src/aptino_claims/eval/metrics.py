"""Metric helpers for the evaluation (pure functions, unit-testable)."""
from __future__ import annotations

import math
from collections import Counter


def confusion_matrix(pairs: list[tuple[str, str]], classes: list[str]) -> dict[str, dict[str, int]]:
    """rows = expected, columns = predicted."""
    counts = Counter(pairs)
    return {e: {p: counts.get((e, p), 0) for p in classes} for e in classes}


def per_class_prf(pairs: list[tuple[str, str]], classes: list[str]) -> dict[str, dict[str, float | int | None]]:
    out = {}
    for c in classes:
        tp = sum(1 for e, p in pairs if e == c and p == c)
        fp = sum(1 for e, p in pairs if e != c and p == c)
        fn = sum(1 for e, p in pairs if e == c and p != c)
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if (prec is not None and rec is not None) else None)
        out[c] = {"support": tp + fn, "precision": prec, "recall": rec, "f1": f1}
    return out


def recall_at_k(ranked: list[str], requirements: dict[str, set[str]], k: int) -> float:
    """Fraction of gold requirements with at least one gold chunk in the top-k."""
    if not requirements:
        return 1.0
    top = set(ranked[:k])
    return sum(1 for chunks in requirements.values() if chunks & top) / len(requirements)


def reciprocal_rank(ranked: list[str], requirements: dict[str, set[str]]) -> float:
    gold_all = set().union(*requirements.values()) if requirements else set()
    for i, cid in enumerate(ranked, start=1):
        if cid in gold_all:
            return 1.0 / i
    return 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q / 100 * len(ordered)) - 1))
    return ordered[idx]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
