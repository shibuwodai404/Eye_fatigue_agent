# Visual Fatigue Agent

多 Agent 协作的视疲劳诊断框架（iScience 投稿导向）。

## 快速开始

```bash
# 1) 进入 conda 环境（首次见下方"环境安装"）
conda activate eye-agent

# 2) 准备环境变量
cp .env.example .env

# 3) 确认 Ollama 在跑、模型已 pull
ollama list   # 期望看到 qwen3.5:35b-a3b（或修改 .env 里的 OLLAMA_MODEL）

# 4) 启动视觉服务（端口 8765）—— 框架版，需要把 Detectron2 推理代码贴进 src/services/vision_api.py 的 TODO
uvicorn src.services.vision_api:app --host 0.0.0.0 --port 8765

# 5) 跑端到端 demo
python -m scripts.demo_end_to_end --video /tmp/mock_video.mp4 --patient-id demo_001
```

## 环境安装（一次性）

```bash
conda create -n eye-agent python=3.11 -y
conda activate eye-agent
pip install -r requirements.txt
```

## 架构

```
[START]
  → IntakeAgent       (主诉 + CVS-Q/OSDI 总分)
  → VisionAgent       (调 FastAPI 视觉服务)
  → FeatureAggregator (时序统计 + 异常检测 + 归一化)
  → RetrievalAgent    (RAG 指南 + 相似病例)
  → DiagnosisAgent    (CoT 推理，3 温度采样)
  → ConsistencyCheck  (多数投票 + 置信度，必要时挂人工)
  → CaseLibraryAgent  (PG/pgvector 入库；缺 PG 时降级 JSONL)
  → ReportAgent       (医生版 + 患者版)
[END]
```

## 关键设计

- 每个 agent 暴露 `run(state) -> state`，可独立调用 / 评估
- 所有 LLM 调用写入 `logs/llm_traces/{patient_id}_{ts}.jsonl`
- 消融开关在 `.env`：`DISABLE_RAG`、`DISABLE_CONSISTENCY`、`DISABLE_CASE_LIBRARY` 等
- 病例库默认 JSONL fallback；装好 PG + pgvector 后改 `CASE_DB_BACKEND=POSTGRES` 即可

## PostgreSQL 切换

第一版用 JSONL 兜底，**未来切到 PG 的步骤**：

```bash
sudo apt install postgresql postgresql-contrib
sudo -u postgres psql -c "CREATE USER eye_agent WITH PASSWORD 'eye_agent';"
sudo -u postgres psql -c "CREATE DATABASE eye_agent OWNER eye_agent;"
sudo -u postgres psql -d eye_agent -c "CREATE EXTENSION vector;"
psql -U eye_agent -d eye_agent -f scripts/init_db.sql
# 然后把 .env 里 CASE_DB_BACKEND 改成 POSTGRES
```

## 目录

见 `REQUIREMENTS.html`（项目需求文档）。
