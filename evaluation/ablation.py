"""消融实验脚手架。

为什么用 subprocess 而不是 importlib.reload？
  各 agent 在 import 时已经 `from src.config import settings`，
  把 settings 单例抓住了。reload 只会更新 src.config.settings，
  agents 手里仍是旧引用 —— 实测会让 ablation 不生效。
  改用 subprocess 隔离：每条 case 起一个新 Python 进程，env 变量
  在子进程 import settings 之前生效，**保证 disable_* 真的落地**。

用法：
  uv run python -m evaluation.ablation \\
      --dataset evaluation/labeled_cases.jsonl \\
      --conditions full no_rag no_consistency no_case_lib

每条 case JSON 行需包含 ground_truth_severity，便于直接算 accuracy。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import typer
from loguru import logger
from rich import print as rprint

app = typer.Typer(add_completion=False)


_CONDITIONS: dict[str, dict[str, str]] = {
    "full": {},
    "no_rag": {"DISABLE_RAG": "true"},
    "no_consistency": {"DISABLE_CONSISTENCY": "true"},
    "no_case_lib": {"DISABLE_CASE_LIBRARY": "true", "DISABLE_SIMILAR_CASES": "true"},
    "no_similar_cases": {"DISABLE_SIMILAR_CASES": "true"},
}


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dataset(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _run_one_subprocess(case: dict, env_overrides: dict[str, str]) -> dict:
    """启动 `uv run python -m scripts.run_single_case`，把 case JSON 喂进 stdin。"""
    env = {**os.environ, **env_overrides}
    cmd = ["uv", "run", "python", "-m", "scripts.run_single_case"]

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        input=json.dumps(case, ensure_ascii=False),
        text=True,
        capture_output=True,
        env=env,
        cwd=str(PROJECT_ROOT),
        timeout=1800,
    )
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        return {
            "patient_id": case.get("patient_id"),
            "predicted_severity": None,
            "predicted_label": None,
            "errors": {"subprocess": proc.stderr[-2000:]},
            "elapsed_s": round(elapsed, 3),
        }
    try:
        last_line = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1]
        return json.loads(last_line)
    except Exception as exc:  # noqa: BLE001
        return {
            "patient_id": case.get("patient_id"),
            "predicted_severity": None,
            "predicted_label": None,
            "errors": {"parse": repr(exc), "stdout": proc.stdout[-1000:]},
            "elapsed_s": round(elapsed, 3),
        }


@app.command()
def main(
    dataset: Path = typer.Option(..., exists=True, help="JSONL，每行 {patient_id, video_path, intake_seed, ground_truth_severity, ...}"),
    conditions: list[str] = typer.Option(["full", "no_rag", "no_consistency"]),
    out_dir: Path = typer.Option(Path("./logs/ablation")),
):
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    cases = _load_dataset(dataset)
    rprint(f"[bold cyan]Dataset:[/] {len(cases)} cases")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    for cond in conditions:
        overrides = _CONDITIONS.get(cond)
        if overrides is None:
            rprint(f"[red]未知 condition:[/] {cond} —— skip")
            continue
        rprint(f"\n[bold yellow]== Condition: {cond} ==[/] overrides={overrides}")

        results: list[dict] = []
        for i, case in enumerate(cases, 1):
            r = _run_one_subprocess(case, overrides)
            r["ground_truth_severity"] = case.get("ground_truth_severity")
            results.append(r)
            rprint(
                f"  [{i:>3}/{len(cases)}] {case.get('patient_id')}  "
                f"pred={r.get('predicted_severity')}  gt={r.get('ground_truth_severity')}  "
                f"{r.get('elapsed_s')}s"
            )

        (out_dir / f"results_{cond}.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        gt_pairs = [
            (r["ground_truth_severity"], r["predicted_severity"])
            for r in results
            if r.get("ground_truth_severity") and r.get("predicted_severity")
        ]
        acc = sum(int(a == b) for a, b in gt_pairs) / len(gt_pairs) if gt_pairs else 0.0
        summary[cond] = {
            "n": len(results),
            "n_scored": len(gt_pairs),
            "accuracy": round(acc, 4),
        }
        rprint(f"   accuracy = {acc:.3f} (n={len(gt_pairs)})")

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rprint(f"\n[green]结果写入 {out_dir}[/]")
    rprint(summary)


if __name__ == "__main__":
    app()
