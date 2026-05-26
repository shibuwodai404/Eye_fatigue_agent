"""LangGraph State 定义。

每个 agent 的产出会以专属字段累加进 state，便于：
- 失败时回溯任一节点的输出
- 消融实验对比不同节点开关下的最终诊断
- LLM trace 完整记录每次调用
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict


# ===== 子结构（dict 形态，避免引入额外的 dataclass / pydantic 约束） =====

class IntakeData(TypedDict, total=False):
    chief_complaint: str                # 患者主诉自由文本
    demographics: dict[str, Any]        # age / sex / occupation / screen_hours_per_day ...
    cvs_q_total: float                  # CVS-Q 总分（外部已统计）
    osdi_total: float                   # OSDI 总分（外部已统计）
    scale_extras: dict[str, Any]        # 其他量表 / 备注
    structured: dict[str, Any]          # IntakeAgent LLM 抽取后的结构化字段


class VisionMetrics(TypedDict, total=False):
    blink_rate_per_min: float
    blink_completeness_ratio: float
    gaze_amplitude_deg: float
    fixation_stability_rms_deg: float
    saccade_count: int
    eye_open_ratio: float
    raw_timeseries_path: str            # 若服务侧落盘
    extras: dict[str, Any]


class AggregatedFeatures(TypedDict, total=False):
    normalized: dict[str, float]        # 各指标 z-score 或 min-max
    anomalies: list[dict[str, Any]]     # [{metric, value, z, severity}, ...]
    summary_stats: dict[str, Any]       # mean/std/p25/p75 等
    composite_score: float              # 综合疲劳分（聚合器算）


class RetrievalHit(TypedDict, total=False):
    source: Literal["guideline", "case"]
    doc_id: str
    title: str
    score: float
    snippet: str
    metadata: dict[str, Any]


class DiagnosisCandidate(TypedDict, total=False):
    temperature: float
    diagnosis_label: str                # 例: "中度视疲劳 / 干眼倾向"
    severity_grade: Literal["none", "mild", "moderate", "severe"]
    reasoning_chain: str                # CoT 文本
    evidence: list[str]                 # 引用的指标 / RAG / 病例 id
    raw_json: dict[str, Any]            # 完整解析结果


class FinalDiagnosis(TypedDict, total=False):
    diagnosis_label: str
    severity_grade: Literal["none", "mild", "moderate", "severe"]
    confidence: float                   # 0..1，由 ConsistencyCheck 算
    reasoning_chain: str                # 选中的那次推理（或合并）
    vote_distribution: dict[str, int]   # 严重度投票分布


class LLMCallRecord(TypedDict, total=False):
    ts: str                             # ISO 时间戳
    agent: str
    model: str
    prompt: str
    output: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    temperature: float
    error: str | None


class AblationFlags(TypedDict, total=False):
    disable_rag: bool
    disable_consistency: bool
    disable_case_library: bool
    disable_similar_cases: bool


class Reports(TypedDict, total=False):
    doctor_md: str
    patient_md: str


# ===== 主 State =====

class GraphState(TypedDict, total=False):
    # ----- 元信息 -----
    patient_id: str
    session_id: str
    timestamp: str
    video_path: str

    # ----- 节点输出 -----
    intake: IntakeData
    vision_metrics: VisionMetrics
    aggregated: AggregatedFeatures
    retrieval: dict[str, list[RetrievalHit]]       # {"guideline": [...], "case": [...]}
    diagnosis_candidates: list[DiagnosisCandidate]
    final_diagnosis: FinalDiagnosis
    requires_human_review: bool

    # ----- 副产物 -----
    case_library_record_id: str | None
    reports: Reports

    # ----- 实验 / 调试 -----
    ablation_flags: AblationFlags
    llm_calls: list[LLMCallRecord]
    errors: dict[str, str]                          # {agent_name: error_str}


def new_state(
    *,
    patient_id: str,
    video_path: str,
    session_id: str | None = None,
    intake_seed: IntakeData | None = None,
) -> GraphState:
    """构造一个干净的 state，调用方传入最小必要字段。"""
    import uuid
    from datetime import datetime, timezone

    return {
        "patient_id": patient_id,
        "session_id": session_id or str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "video_path": video_path,
        "intake": intake_seed or {},
        "vision_metrics": {},
        "aggregated": {},
        "retrieval": {"guideline": [], "case": []},
        "diagnosis_candidates": [],
        "final_diagnosis": {},
        "requires_human_review": False,
        "case_library_record_id": None,
        "reports": {},
        "ablation_flags": {},
        "llm_calls": [],
        "errors": {},
    }
