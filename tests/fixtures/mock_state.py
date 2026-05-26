"""测试用 mock state 构造器。"""
from __future__ import annotations

from src.state import GraphState, new_state


def make_mock_state(*, patient_id: str = "test_001", video_path: str = "/tmp/mock.mp4") -> GraphState:
    s = new_state(
        patient_id=patient_id,
        video_path=video_path,
        intake_seed={
            "chief_complaint": "近一周用电脑后眼睛干涩、视物模糊",
            "demographics": {"age": 30, "sex": "F", "occupation": "设计师", "screen_hours_per_day": 9},
            "cvs_q_total": 12,
            "osdi_total": 20,
        },
    )
    s["vision_metrics"] = {
        "blink_rate_per_min": 9.0,
        "blink_completeness_ratio": 0.65,
        "gaze_amplitude_deg": 5.5,
        "fixation_stability_rms_deg": 0.58,
        "saccade_count": 95,
        "eye_open_ratio": 0.86,
        "extras": {"mock": True},
    }
    return s
