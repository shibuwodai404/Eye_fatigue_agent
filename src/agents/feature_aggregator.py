"""FeatureAggregator —— 程序化处理：归一化、异常检测、合成综合分。

阈值取自文献的粗略参考（仅占位，后续需要用真实病例标定）。
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from src.state import GraphState

AGENT = "feature_aggregator"


# 健康成人参考区间 (mu, sigma)，仅占位；TODO: 用真实病例标定
_REFERENCE: dict[str, tuple[float, float]] = {
    "blink_rate_per_min": (15.0, 4.0),
    "blink_completeness_ratio": (0.85, 0.08),
    "gaze_amplitude_deg": (3.0, 1.2),
    "fixation_stability_rms_deg": (0.30, 0.10),
    "saccade_count": (60.0, 20.0),
    "eye_open_ratio": (0.92, 0.04),
}

# 综合疲劳分权重（越高 = 越不健康方向贡献越大）
_WEIGHTS: dict[str, float] = {
    "blink_rate_per_min": -0.20,
    "blink_completeness_ratio": -0.25,
    "gaze_amplitude_deg": 0.10,
    "fixation_stability_rms_deg": 0.25,
    "saccade_count": 0.05,
    "eye_open_ratio": -0.15,
}


def _zscore(value: float, mu: float, sigma: float) -> float:
    return (value - mu) / (sigma + 1e-9)


def run(state: GraphState) -> GraphState:
    metrics = state.get("vision_metrics") or {}
    if not metrics:
        msg = "vision_metrics 为空，无法聚合"
        logger.error(f"[aggregator] {msg}")
        state.setdefault("errors", {})[AGENT] = msg
        return state

    normalized: dict[str, float] = {}
    anomalies: list[dict[str, Any]] = []
    composite = 0.0

    for key, (mu, sigma) in _REFERENCE.items():
        val = metrics.get(key)
        if val is None:
            continue
        z = _zscore(float(val), mu, sigma)
        normalized[key] = round(z, 3)

        # 综合分 = sum(weight * z)，权重符号反映"指标变大/变小"是否更不健康
        composite += _WEIGHTS.get(key, 0.0) * z

        if abs(z) >= 2.0:
            sev = "severe"
        elif abs(z) >= 1.0:
            sev = "moderate"
        elif abs(z) >= 0.5:
            sev = "mild"
        else:
            sev = "normal"
        if sev != "normal":
            anomalies.append(
                {
                    "metric": key,
                    "value": float(val),
                    "z": round(z, 3),
                    "severity": sev,
                }
            )

    summary_stats = {
        k: round(float(v), 3)
        for k, v in metrics.items()
        if isinstance(v, (int, float))
    }

    aggregated = {
        "normalized": normalized,
        "anomalies": anomalies,
        "summary_stats": summary_stats,
        "composite_score": round(composite, 3),
    }
    logger.info(
        f"[aggregator] composite={aggregated['composite_score']} "
        f"anomalies={len(anomalies)}"
    )

    new_state = dict(state)
    new_state["aggregated"] = aggregated
    return new_state  # type: ignore[return-value]
