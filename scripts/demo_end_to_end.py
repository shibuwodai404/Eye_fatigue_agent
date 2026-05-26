"""端到端 demo：用 mock 视频跑一次，输出报告。

用法：
  uv run python -m scripts.demo_end_to_end --video /tmp/mock_video.mp4
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from loguru import logger
from rich import print as rprint

from src.config import settings
from src.graph import compiled_graph
from src.llm import warmup
from src.state import new_state

app = typer.Typer(add_completion=False)


_MOCK_INTAKE = {
    "chief_complaint": (
        "近两周长时间用电脑后眼睛干涩、刺痛，有时下班后视物模糊；"
        "晨起略有畏光，每日屏幕使用约 10 小时，工作以编程为主。"
    ),
    "demographics": {
        "age": 29,
        "sex": "M",
        "occupation": "软件工程师",
        "screen_hours_per_day": 10,
    },
    "cvs_q_total": 14,   # 中
    "osdi_total": 25,    # 中
}


@app.command()
def main(
    video: str = typer.Option("/tmp/mock_video.mp4", help="mock 视频路径（不存在也无所谓，视觉服务未起会返回 mock 指标）"),
    patient_id: str = typer.Option("demo_001"),
    skip_warmup: bool = typer.Option(False, "--skip-warmup"),
):
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)

    if not skip_warmup:
        warmup(settings.ollama_model)

    state = new_state(
        patient_id=patient_id,
        video_path=video,
        intake_seed=_MOCK_INTAKE,
    )

    rprint("[bold cyan]=== Demo start ===[/]")
    rprint(
        {
            "ollama_model": settings.ollama_model,
            "case_db_backend": settings.case_db_backend,
            "ablation": settings.ablation.model_dump(),
        }
    )

    final = compiled_graph.invoke(state)

    out_dir = settings.log_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    sid = final["session_id"]
    (out_dir / f"{patient_id}_{sid}.state.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    if (r := final.get("reports") or {}).get("doctor_md"):
        (out_dir / f"{patient_id}_{sid}.doctor.md").write_text(r["doctor_md"], encoding="utf-8")
    if (r := final.get("reports") or {}).get("patient_md"):
        (out_dir / f"{patient_id}_{sid}.patient.md").write_text(r["patient_md"], encoding="utf-8")

    rprint("[bold green]=== Demo done ===[/]")
    rprint(
        {
            "final_diagnosis": final.get("final_diagnosis"),
            "requires_human_review": final.get("requires_human_review"),
            "case_id": final.get("case_library_record_id"),
            "n_llm_calls": len(final.get("llm_calls") or []),
            "errors": final.get("errors") or {},
        }
    )
    rprint(f"[dim]输出目录：{out_dir}[/]")


if __name__ == "__main__":
    app()
