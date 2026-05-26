"""诊断评估指标：准确率、F1、Cohen's κ。

输入是两份等长 list[str]（严重度标签）或两份 dict（label + severity）。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


_SEV_ORDER = ["none", "mild", "moderate", "severe"]


def severity_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    average: str = "macro",
) -> dict[str, float | list]:
    """严重度分级评估。"""
    labels = _SEV_ORDER
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average, labels=labels, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "cohen_kappa_quadratic": float(kappa),
        "confusion_matrix": cm,
        "labels": labels,
        "per_class": report,
    }


def diagnosis_label_match_rate(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """诊断标签精确匹配率（粗粒度）。"""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    return float((yt == yp).mean()) if len(yt) else 0.0
