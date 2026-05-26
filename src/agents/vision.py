"""VisionAgent —— 调用 FastAPI 视觉服务，把行为指标写入 state。"""
from __future__ import annotations

from loguru import logger

from src.state import GraphState
from src.tools import vision_service_client as vision

AGENT = "vision"


def run(state: GraphState) -> GraphState:
    video_path = state.get("video_path")
    if not video_path:
        msg = "video_path 未设置"
        logger.error(f"[vision] {msg}")
        state.setdefault("errors", {})[AGENT] = msg
        return state

    if not vision.health():
        msg = f"vision API 未就绪 @ {vision.settings.vision_api_base_url}"
        logger.warning(f"[vision] {msg}; 返回 mock 指标，便于框架走通")
        metrics = {
            "blink_rate_per_min": 9.5,
            "blink_completeness_ratio": 0.68,
            "gaze_amplitude_deg": 4.2,
            "fixation_stability_rms_deg": 0.55,
            "saccade_count": 76,
            "eye_open_ratio": 0.89,
            "extras": {"mock": True, "reason": msg},
        }
        new_state = dict(state)
        new_state["vision_metrics"] = metrics
        return new_state  # type: ignore[return-value]

    try:
        metrics = vision.analyze_video(video_path, patient_id=state.get("patient_id"))
    except Exception as exc:  # noqa: BLE001
        logger.exception("[vision] 调用视觉服务失败")
        state.setdefault("errors", {})[AGENT] = repr(exc)
        return state

    new_state = dict(state)
    new_state["vision_metrics"] = metrics
    return new_state  # type: ignore[return-value]
