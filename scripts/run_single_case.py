"""单条 case 运行驱动 —— 供 evaluation/ablation.py 用 subprocess 隔离调用。

输入：
  - 通过 stdin 读取 JSON：{patient_id, video_path, intake_seed, vision_metrics_override?}
  - 环境变量按调用方设置（DISABLE_* 等）

输出：
  - 一行 JSON 到 stdout：{patient_id, predicted_severity, predicted_label, confidence,
                          requires_human_review, n_llm_calls, errors, elapsed_s}

例：
  echo '{"patient_id":"x","video_path":"/tmp/m.mp4","intake_seed":{...}}' | \\
      DISABLE_RAG=true uv run python -m scripts.run_single_case
"""
from __future__ import annotations

import json
import sys
import time


def main() -> int:
    payload = json.loads(sys.stdin.read())

    # 必须在读取 env 之后再 import，确保 settings 拿到当前 env
    from src.graph import compiled_graph
    from src.state import new_state

    state = new_state(
        patient_id=payload["patient_id"],
        video_path=payload.get("video_path", "/tmp/mock.mp4"),
        intake_seed=payload.get("intake_seed", {}),
    )
    if "vision_metrics_override" in payload:
        state["vision_metrics"] = payload["vision_metrics_override"]

    t0 = time.perf_counter()
    out = compiled_graph.invoke(state)
    elapsed = time.perf_counter() - t0

    final = out.get("final_diagnosis") or {}
    result = {
        "patient_id": payload["patient_id"],
        "predicted_severity": final.get("severity_grade"),
        "predicted_label": final.get("diagnosis_label"),
        "confidence": final.get("confidence"),
        "requires_human_review": out.get("requires_human_review"),
        "n_llm_calls": len(out.get("llm_calls") or []),
        "errors": out.get("errors") or {},
        "elapsed_s": round(elapsed, 3),
    }
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
