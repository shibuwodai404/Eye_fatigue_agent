"""每个 agent 的独立 smoke test —— 消融实验对独立可调用的硬要求。

对依赖外部服务（Ollama / 视觉服务 / embedding / DB）的 agent，
默认 skip，仅在显式打开 RUN_EXTERNAL_AGENT_TESTS=1 时跑。
"""
from __future__ import annotations

import os

import pytest

from src.agents import (
    case_library,
    consistency,
    feature_aggregator,
    vision,
)
from tests.fixtures.mock_state import make_mock_state


_EXTERNAL = os.environ.get("RUN_EXTERNAL_AGENT_TESTS") == "1"
skip_external = pytest.mark.skipif(not _EXTERNAL, reason="需要外部服务，设 RUN_EXTERNAL_AGENT_TESTS=1 启用")


# ---------- 纯本地 agent：直接跑 ----------

def test_feature_aggregator_runs():
    s = make_mock_state()
    out = feature_aggregator.run(s)
    agg = out["aggregated"]
    assert "normalized" in agg and "anomalies" in agg and "composite_score" in agg
    # 该 mock 数据明显偏离健康参考，应该有异常项
    assert len(agg["anomalies"]) >= 1


def test_vision_returns_mock_when_service_down():
    """视觉服务未起时 vision agent 应返回 mock 指标而不是崩。"""
    s = make_mock_state()
    s["vision_metrics"] = {}
    out = vision.run(s)
    assert "vision_metrics" in out
    assert out["vision_metrics"]


def test_consistency_majority_vote():
    s = make_mock_state()
    s["diagnosis_candidates"] = [
        {"temperature": 0.3, "diagnosis_label": "中度视疲劳", "severity_grade": "moderate",
         "reasoning_chain": "ra", "evidence": [], "raw_json": {}},
        {"temperature": 0.5, "diagnosis_label": "中度视疲劳", "severity_grade": "moderate",
         "reasoning_chain": "rb", "evidence": [], "raw_json": {}},
        {"temperature": 0.7, "diagnosis_label": "轻度视疲劳", "severity_grade": "mild",
         "reasoning_chain": "rc", "evidence": [], "raw_json": {}},
    ]
    out = consistency.run(s)
    assert out["final_diagnosis"]["severity_grade"] == "moderate"
    assert out["final_diagnosis"]["confidence"] > 0.5
    assert out["requires_human_review"] is False


def test_consistency_human_review_when_spread_big():
    s = make_mock_state()
    s["diagnosis_candidates"] = [
        {"temperature": 0.3, "diagnosis_label": "轻度", "severity_grade": "mild",
         "reasoning_chain": "", "evidence": [], "raw_json": {}},
        {"temperature": 0.5, "diagnosis_label": "中度", "severity_grade": "moderate",
         "reasoning_chain": "", "evidence": [], "raw_json": {}},
        {"temperature": 0.7, "diagnosis_label": "重度", "severity_grade": "severe",
         "reasoning_chain": "", "evidence": [], "raw_json": {}},
    ]
    out = consistency.run(s)
    assert out["requires_human_review"] is True


# ---------- 依赖外部服务的 agent ----------

@skip_external
def test_intake_with_real_llm():
    from src.agents import intake
    s = make_mock_state()
    out = intake.run(s)
    assert "structured" in out["intake"]


@skip_external
def test_retrieval_with_real_index():
    from src.agents import retrieval
    s = make_mock_state()
    # feature_aggregator 先跑一遍，把 anomalies 填进去
    s = feature_aggregator.run(s)
    out = retrieval.run(s)
    assert "guideline" in out["retrieval"]
    assert "case" in out["retrieval"]


@skip_external
def test_case_library_insert(tmp_path, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "case_db_jsonl_path", tmp_path / "cases.jsonl")
    s = make_mock_state()
    s["final_diagnosis"] = {
        "diagnosis_label": "中度视疲劳",
        "severity_grade": "moderate",
        "confidence": 0.67,
        "reasoning_chain": "...",
        "vote_distribution": {"moderate": 2, "mild": 1},
    }
    out = case_library.run(s)
    assert out["case_library_record_id"]
