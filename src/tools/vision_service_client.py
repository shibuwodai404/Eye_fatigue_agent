"""调用 FastAPI 视觉服务获取行为指标。"""
from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from src.config import settings


def analyze_video(video_path: str, *, patient_id: str | None = None) -> dict[str, Any]:
    """POST /analyze，返回视觉服务的 JSON 输出。

    服务端约定见 src/services/vision_api.py —— 返回结构示例：
    {
      "blink_rate_per_min": 12.3,
      "blink_completeness_ratio": 0.78,
      "gaze_amplitude_deg": 5.6,
      "fixation_stability_rms_deg": 0.42,
      "saccade_count": 88,
      "eye_open_ratio": 0.93,
      "raw_timeseries_path": "/tmp/xxx.parquet",
      "extras": {...}
    }
    """
    url = f"{settings.vision_api_base_url.rstrip('/')}/analyze"
    payload: dict[str, Any] = {"video_path": video_path}
    if patient_id is not None:
        payload["patient_id"] = patient_id

    logger.info(f"[vision_client] POST {url} video={video_path}")
    with httpx.Client(timeout=settings.vision_api_timeout_seconds) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return r.json()


def health() -> bool:
    """GET /health；视觉服务未起时返回 False，方便上游降级。"""
    url = f"{settings.vision_api_base_url.rstrip('/')}/health"
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(url)
            return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False
