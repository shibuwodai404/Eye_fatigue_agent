"""state.new_state 基本字段检查。"""
from __future__ import annotations

from src.state import new_state


def test_new_state_minimum_fields():
    s = new_state(patient_id="p1", video_path="/x.mp4")
    assert s["patient_id"] == "p1"
    assert s["video_path"] == "/x.mp4"
    assert s["session_id"]
    assert s["timestamp"]
    assert s["intake"] == {}
    assert s["retrieval"] == {"guideline": [], "case": []}
    assert s["llm_calls"] == []
    assert s["errors"] == {}


def test_new_state_with_intake_seed():
    s = new_state(
        patient_id="p2",
        video_path="/y.mp4",
        intake_seed={"chief_complaint": "干涩", "cvs_q_total": 10, "osdi_total": 15},
    )
    assert s["intake"]["chief_complaint"] == "干涩"
    assert s["intake"]["cvs_q_total"] == 10
