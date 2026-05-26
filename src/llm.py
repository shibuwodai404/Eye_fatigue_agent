"""Ollama 调用 + LLM trace 统一封装。

约束：
- 每次调用必须记录到 state["llm_calls"] 和 logs/llm_traces/{patient_id}_{ts}.jsonl
- 默认 JSON 模式（format="json"），由调用方在 prompt 中说明字段结构
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from src.config import settings


def _trace_path(patient_id: str, session_id: str) -> Path:
    fname = f"{patient_id}_{session_id}.jsonl"
    return settings.llm_trace_dir / fname


def _append_trace(patient_id: str, session_id: str, record: dict[str, Any]) -> None:
    p = _trace_path(patient_id, session_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def chat(
    *,
    agent: str,
    system: str,
    user: str,
    state: dict[str, Any],
    temperature: float | None = None,
    model: str | None = None,
    format_json: bool = True,
) -> str:
    """调用 Ollama /api/chat，返回 assistant content（字符串）。

    所有调用都会：
    1) 追加到 state["llm_calls"]
    2) 追加到 logs/llm_traces/{patient_id}_{session_id}.jsonl
    """
    model = model or settings.ollama_model
    temperature = temperature if temperature is not None else settings.default_temperature

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    if format_json:
        payload["format"] = "json"

    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    t0 = time.perf_counter()
    err: str | None = None
    output_text = ""
    prompt_tokens = 0
    completion_tokens = 0

    try:
        with httpx.Client(timeout=settings.llm_timeout_seconds) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            output_text = (data.get("message") or {}).get("content", "") or ""
            prompt_tokens = data.get("prompt_eval_count", 0) or 0
            completion_tokens = data.get("eval_count", 0) or 0
    except Exception as exc:  # noqa: BLE001
        err = repr(exc)
        logger.error(f"[llm.chat] {agent} failed: {err}")

    latency_ms = (time.perf_counter() - t0) * 1000.0

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "model": model,
        "prompt": f"<<SYSTEM>>\n{system}\n<<USER>>\n{user}",
        "output": output_text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms": round(latency_ms, 2),
        "temperature": temperature,
        "error": err,
    }
    state.setdefault("llm_calls", []).append(record)
    _append_trace(
        state.get("patient_id", "anon"),
        state.get("session_id", "nosession"),
        record,
    )

    if err:
        raise RuntimeError(f"LLM call failed in {agent}: {err}")
    return output_text


def parse_json_loose(text: str) -> dict[str, Any]:
    """Ollama 在 format=json 下偶尔返回带前后噪声的字符串，宽松解析。"""
    text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 兜底：截取首个 { 到最后一个 }
        l = text.find("{")
        r = text.rfind("}")
        if l != -1 and r != -1 and r > l:
            try:
                return json.loads(text[l : r + 1])
            except json.JSONDecodeError:
                pass
    logger.warning(f"[llm.parse_json_loose] failed to parse: {text[:200]}...")
    return {"_raw": text}


def warmup(model: str | None = None) -> None:
    """预热模型 —— MoE 首次加载较慢，demo / 服务启动时调一次。"""
    model = model or settings.ollama_model
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    logger.info(f"[llm.warmup] preloading {model} ...")
    try:
        with httpx.Client(timeout=settings.llm_timeout_seconds) as client:
            client.post(
                url,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 1},
                },
            )
        logger.info(f"[llm.warmup] {model} ready")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[llm.warmup] failed: {exc!r}")


def prompt(name: str) -> str:
    """读取 src/prompts/<name>.md 的内容。"""
    p = Path(__file__).resolve().parent / "prompts" / f"{name}.md"
    return p.read_text(encoding="utf-8")
