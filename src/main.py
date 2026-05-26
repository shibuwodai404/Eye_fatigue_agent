"""CLI 入口 —— 跑一次完整流程。

示例：
  uv run python -m src.main \\
      --video /tmp/mock_video.mp4 \\
      --patient-id demo_001 \\
      --chief-complaint "近一周用电脑后眼睛干涩、有刺痛" \\
      --cvs-q 14 --osdi 24
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

app = typer.Typer(add_completion=False, help="视疲劳诊断 agent CLI")


@app.command()
def run(
    video: str = typer.Option(..., help="视频文件路径（mp4）"),
    patient_id: str = typer.Option(..., "--patient-id", help="患者 ID"),
    chief_complaint: str = typer.Option("", help="主诉自由文本（中文）"),
    cvs_q: float = typer.Option(None, "--cvs-q", help="CVS-Q 总分"),
    osdi: float = typer.Option(None, "--osdi", help="OSDI 总分"),
    age: int = typer.Option(None),
    sex: str = typer.Option(None),
    occupation: str = typer.Option(None),
    screen_hours: float = typer.Option(None, "--screen-hours"),
    skip_warmup: bool = typer.Option(False, "--skip-warmup", help="跳过 Ollama 预热"),
    out_dir: Path = typer.Option(Path("./logs/reports"), help="报告输出目录"),
):
    """跑一次完整诊断流程。"""
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)

    if not skip_warmup:
        warmup(settings.ollama_model)

    intake_seed = {
        "chief_complaint": chief_complaint,
        "demographics": {
            k: v
            for k, v in {
                "age": age,
                "sex": sex,
                "occupation": occupation,
                "screen_hours_per_day": screen_hours,
            }.items()
            if v is not None
        },
        "cvs_q_total": cvs_q,
        "osdi_total": osdi,
    }

    state = new_state(patient_id=patient_id, video_path=video, intake_seed=intake_seed)
    rprint(f"[bold cyan]>> 启动 graph[/]  patient_id={patient_id}")
    final_state = compiled_graph.invoke(state)

    # 落盘报告 + state
    out_dir.mkdir(parents=True, exist_ok=True)
    sid = final_state.get("session_id", "nosession")
    base = out_dir / f"{patient_id}_{sid}"
    base.with_suffix(".state.json").write_text(
        json.dumps(final_state, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    reports = final_state.get("reports") or {}
    if reports.get("doctor_md"):
        (base.parent / f"{patient_id}_{sid}.doctor.md").write_text(
            reports["doctor_md"], encoding="utf-8"
        )
    if reports.get("patient_md"):
        (base.parent / f"{patient_id}_{sid}.patient.md").write_text(
            reports["patient_md"], encoding="utf-8"
        )

    rprint("[bold green]>> 完成[/]")
    rprint(
        {
            "final_diagnosis": final_state.get("final_diagnosis"),
            "requires_human_review": final_state.get("requires_human_review"),
            "case_id": final_state.get("case_library_record_id"),
            "errors": final_state.get("errors"),
            "n_llm_calls": len(final_state.get("llm_calls") or []),
        }
    )
    rprint(f"[dim]报告 / state 已写入 {out_dir}[/]")


if __name__ == "__main__":
    app()
