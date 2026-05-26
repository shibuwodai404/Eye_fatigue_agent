# Eye-agent · Visual Fatigue Diagnosis with Multi-Agent LLM

> 多 Agent 协作的视疲劳（visual fatigue / asthenopia）诊断研究框架 ——
> 视觉模型 + 本地 LLM + RAG 指南 + 自增长病例库，端到端在本地运行，数据不出域。

[![Status: Research](https://img.shields.io/badge/status-research-orange)](#status)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-7c3aed)](https://github.com/langchain-ai/langgraph)
[![LLM Backend: Ollama](https://img.shields.io/badge/LLM-Ollama-000)](https://ollama.com/)

---

## ✨ What is this?

一个面向**临床场景**的研究原型：把一个已训好的眼部行为视觉模型（基于 Detectron2 的眨眼 / 眼动 / 注视稳定性指标）作为 **tool** 暴露给 LLM agent，agent 综合：

- 视觉行为指标
- 患者主诉与量表分（CVS-Q / OSDI）
- RAG 检索到的临床指南
- 历史相似病例（pgvector 检索）

→ 输出**可解释**的诊断标签 + 严重度分级 + 医生 / 患者双版本报告。

每一次诊断会结构化入库，**系统越用越聪明**；每一次 LLM 调用全量 trace，**支持消融实验与方法学复核**。

## 🎯 Key Features

- **8 节点 LangGraph 顺序流水线** —— Intake → Vision → Aggregator → Retrieval → Diagnosis → Consistency → CaseLibrary → Report
- **Self-consistency 投票** —— DiagnosisAgent 在 3 个 temperature (0.3 / 0.5 / 0.7) 下并行采样，ConsistencyCheck 用多数投票 + 置信度评估，分歧跨级自动触发 `requires_human_review`
- **Tool-augmented diagnosis** —— Detectron2 推理通过独立 FastAPI 服务暴露（端口 8765），agent 工程与视觉依赖完全解耦
- **可消融** —— `.env` 一键开关：`DISABLE_RAG` / `DISABLE_CONSISTENCY` / `DISABLE_CASE_LIBRARY` / `DISABLE_SIMILAR_CASES`
- **可追溯** —— 每次 LLM 调用记录 prompt / output / tokens / latency / temperature 到 `logs/llm_traces/*.jsonl`
- **降级 friendly** —— 病例库默认 JSONL fallback；装好 PostgreSQL + pgvector 后改 `CASE_DB_BACKEND=POSTGRES` 无缝切换
- **本地隐私优先** —— 所有计算在本机完成，无云端依赖，适合医疗合规场景

## 🏗 Architecture

```
[START]
  → IntakeAgent       (主诉 + CVS-Q/OSDI + 人口学 → LLM 抽取结构化字段)
  → VisionAgent       (HTTP 调 FastAPI 视觉服务，拉取行为指标)
  → FeatureAggregator (z-score 归一化 + 异常检测 + 综合疲劳分)
  → RetrievalAgent    (LlamaIndex + BGE-M3 RAG + pgvector 找相似病例)
  → DiagnosisAgent    (CoT 推理 × 3 temperatures)
  → ConsistencyCheck  (多数投票 + 置信度，必要时挂人工)
  → CaseLibraryAgent  (PG/pgvector 入库；缺 PG 时降级 JSONL)
  → ReportAgent       (医生版 + 患者版 Markdown 报告)
[END]
```

详细的设计动机、状态 schema、关键取舍见 [`OVERVIEW.html`](./OVERVIEW.html)。

## 🛠 Tech Stack

| 层 | 选型 |
|---|---|
| Agent 编排 | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM 后端 | [Ollama](https://ollama.com/) + `langchain-ollama` |
| 视觉模型 | Detectron2（独立 FastAPI 服务，端口 8765） |
| RAG | [LlamaIndex](https://www.llamaindex.ai/) + [BGE-M3](https://huggingface.co/BAAI/bge-m3) embedding |
| 病例库 | PostgreSQL + [pgvector](https://github.com/pgvector/pgvector)（或 JSONL fallback） |
| 配置 | `pydantic-settings` + `.env` |
| 日志 | `loguru` + JSONL trace |
| CLI | `typer` + `rich` |

## 🚀 Quickstart

### 1) 环境

```bash
conda create -n eye-agent python=3.11 -y
conda activate eye-agent
pip install -r requirements.txt
cp .env.example .env   # 按需修改 OLLAMA_MODEL 等
```

### 2) 准备本地 Ollama

```bash
# 确认服务在跑
ollama list

# 拉所需模型（示例，按 .env 里 OLLAMA_MODEL 来）
ollama pull qwen3.5:35b-a3b
```

### 3) 启动视觉服务

```bash
uvicorn src.services.vision_api:app --host 0.0.0.0 --port 8765
```

> ⚠️ `src/services/vision_api.py` 中保留了 Detectron2 推理代码的 `TODO` 占位 ——
> 第一版用 mock 指标也能跑通端到端流程；接入真实视觉模型后替换该处即可。

### 4) 端到端 demo

```bash
python -m scripts.demo_end_to_end \
    --video /tmp/mock_video.mp4 \
    --patient-id demo_001
```

产物位于 `logs/`：
- `logs/reports/{patient_id}_{session_id}.{state.json,doctor.md,patient.md}`
- `logs/llm_traces/{patient_id}_{session_id}.jsonl`

## 📂 Project Structure

```
.
├── src/
│   ├── graph.py              # LangGraph 主图编排
│   ├── state.py              # GraphState TypedDict
│   ├── llm.py                # Ollama 统一封装 + trace
│   ├── config.py             # pydantic-settings
│   ├── main.py               # CLI 入口
│   ├── agents/               # 8 个 agent
│   ├── tools/                # 视觉服务客户端 / 病例库 / RAG
│   ├── services/vision_api.py  # FastAPI 视觉服务（含 Detectron2 TODO）
│   └── prompts/              # 所有 system prompt 独立成文件
├── scripts/
│   ├── demo_end_to_end.py    # mock 数据端到端 demo
│   ├── run_single_case.py
│   ├── ingest_knowledge.py   # RAG 知识库灌入
│   ├── init_db.sql           # PostgreSQL schema
│   └── knowledge_docs/       # 占位指南文档
├── evaluation/
│   ├── metrics.py            # accuracy / F1 / Cohen κ
│   └── ablation.py           # 消融实验脚手架
├── tests/
│   ├── test_each_agent.py    # 每个 agent 独立测试
│   ├── test_state.py
│   └── fixtures/
├── pyproject.toml
├── requirements.txt
├── .env.example
├── README.md                 # 本文件（公开版）
├── README.dev.md             # 内部 / 开发者说明
├── OVERVIEW.html             # 项目全景梳理（motivation + 框架）
└── REQUIREMENTS.html         # 原始需求文档
```

## 🧪 Evaluation & Ablation

```bash
# 在 .env 里翻转单个开关，然后重跑同一批病例对比
DISABLE_RAG=true python -m scripts.run_single_case --case-id case_001
DISABLE_CONSISTENCY=true python -m scripts.run_single_case --case-id case_001
```

或用 `evaluation/ablation.py` 的脚手架批量跑组合。

## 🗄 升级到 PostgreSQL

第一版用 JSONL 兜底足够实验；规模化后：

```bash
sudo apt install postgresql postgresql-contrib
sudo -u postgres psql -c "CREATE USER eye_agent WITH PASSWORD 'eye_agent';"
sudo -u postgres psql -c "CREATE DATABASE eye_agent OWNER eye_agent;"
sudo -u postgres psql -d eye_agent -c "CREATE EXTENSION vector;"
psql -U eye_agent -d eye_agent -f scripts/init_db.sql
# 然后把 .env 里 CASE_DB_BACKEND 改成 POSTGRES
```

## 📚 Status

<a id="status"></a>

| 模块 | 状态 |
|---|---|
| LangGraph 编排 + 8 agent | ✅ 跑通 mock 端到端 |
| Self-consistency + 置信度 | ✅ |
| RAG (LlamaIndex + BGE-M3) | ✅ 占位指南可用 |
| 病例库 JSONL backend | ✅ |
| 病例库 PostgreSQL backend | ⚙️ schema 已就绪，等数据 |
| Detectron2 视觉模型接入 | 🚧 接口占位，等贴入推理代码 |
| 消融实验批跑 | 🚧 脚手架已搭 |
| 临床数据评估 | ⏳ 待数据 |

## 📜 Disclaimer

**This repository is research code, not a medical device.**
Outputs are intended for academic / methodology exploration only and **must not** be used as a basis for clinical decisions without qualified ophthalmologist review.

本项目为研究原型，**不构成医疗诊断建议**。任何基于本系统输出的临床决策，须由具备资质的眼科医生复核。

## 🤝 Citation

如该框架对你的研究有帮助，欢迎引用（论文 BibTeX 发表后补充）：

```bibtex
@misc{eye_agent_2026,
  title  = {Eye-agent: A Multi-Agent LLM Framework for Visual Fatigue Diagnosis},
  author = {Li, Guangyu and ...},
  year   = {2026},
  url    = {https://github.com/shibuwodai404/Eye_fatigue_agent}
}
```

## 📄 License

[MIT](./LICENSE) © 2026 shibuwodai404
