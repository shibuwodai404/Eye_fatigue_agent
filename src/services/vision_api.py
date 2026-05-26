"""FastAPI 包装的视觉推理服务。

# ============================================================
# !!! 关键 TODO !!!
# ------------------------------------------------------------
# 本文件只提供 HTTP 骨架，把现有的 Detectron2 推理代码贴进
# `_run_detectron2_inference(...)` 函数体内即可。
# 不要在这里写 Detectron2 模型加载 / 训练 / 推理逻辑 ——
# 这部分由用户的现成代码负责。
# ============================================================

启动：
  uv run uvicorn src.services.vision_api:app --host 0.0.0.0 --port 8765
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

app = FastAPI(title="Eye Behavior Vision API", version="0.1.0")


class AnalyzeRequest(BaseModel):
    video_path: str
    patient_id: str | None = None
    extras: dict[str, Any] | None = None


class AnalyzeResponse(BaseModel):
    blink_rate_per_min: float
    blink_completeness_ratio: float
    gaze_amplitude_deg: float
    fixation_stability_rms_deg: float
    saccade_count: int
    eye_open_ratio: float
    raw_timeseries_path: str | None = None
    extras: dict[str, Any] = {}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "vision_api"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    p = Path(req.video_path)
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"video_path 不存在: {req.video_path}")
    logger.info(f"[vision_api] analyze video={p} patient={req.patient_id}")

    metrics = _run_detectron2_inference(str(p), patient_id=req.patient_id)
    return AnalyzeResponse(**metrics)


# ============================================================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# TODO(user): 把你现有的 Detectron2 推理 + 行为指标计算贴这里。
# 输入：video_path (str)，可选 patient_id
# 输出：dict，字段对齐 AnalyzeResponse
# ============================================================
def _run_detectron2_inference(video_path: str, *, patient_id: str | None = None) -> dict[str, Any]:
    """占位实现：返回一个随机但合理的指标，方便框架先跑通。
    替换为真实推理时，请保留返回 dict 的 schema（与 AnalyzeResponse 对齐）。
    """
    import random

    random.seed(hash((video_path, patient_id)) & 0xFFFFFFFF)
    t0 = time.perf_counter()
    metrics = {
        "blink_rate_per_min": round(random.uniform(6.0, 22.0), 2),
        "blink_completeness_ratio": round(random.uniform(0.55, 0.95), 3),
        "gaze_amplitude_deg": round(random.uniform(2.0, 7.0), 2),
        "fixation_stability_rms_deg": round(random.uniform(0.20, 0.80), 3),
        "saccade_count": random.randint(40, 120),
        "eye_open_ratio": round(random.uniform(0.80, 0.97), 3),
        "raw_timeseries_path": None,
        "extras": {
            "placeholder": True,
            "video_path": video_path,
            "compute_ms": round((time.perf_counter() - t0) * 1000, 2),
            "note": "替换 _run_detectron2_inference 为真实推理后删除 placeholder=True",
        },
    }
    return metrics


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8765))
    uvicorn.run(app, host="0.0.0.0", port=port)
