-- ============================================================
-- 视疲劳诊断 - 病例库 schema
-- 使用前：
--   CREATE DATABASE eye_agent;
--   \c eye_agent
--   CREATE EXTENSION IF NOT EXISTS vector;
--   \i scripts/init_db.sql
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS cases (
    id                   UUID PRIMARY KEY,
    patient_id           TEXT NOT NULL,
    captured_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    chief_complaint      TEXT,
    scale_scores         JSONB,                       -- CVS-Q / OSDI / 其他
    vision_metrics       JSONB,                       -- 行为指标
    final_diagnosis      TEXT,
    severity_grade       TEXT CHECK (severity_grade IN ('none','mild','moderate','severe')),
    reasoning_chain      TEXT,                        -- agent 推理链全文
    doctor_review        TEXT,                        -- 医生复核（可空，后续监督学习用）
    video_path           TEXT,
    symptom_embedding    vector(1024),                -- BGE-M3 主诉 embedding
    multimodal_embedding vector(1024),                -- 主诉 + 指标拼接的 embedding（相似检索用）
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 索引：常用查询
CREATE INDEX IF NOT EXISTS idx_cases_patient_id   ON cases (patient_id);
CREATE INDEX IF NOT EXISTS idx_cases_captured_at  ON cases (captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_cases_severity     ON cases (severity_grade);

-- 向量索引：相似病例检索（IVFFlat / HNSW 任选；HNSW 召回 / 速度更好）
CREATE INDEX IF NOT EXISTS idx_cases_multimodal_hnsw
    ON cases USING hnsw (multimodal_embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_cases_symptom_hnsw
    ON cases USING hnsw (symptom_embedding vector_cosine_ops);

-- 自动更新 updated_at
CREATE OR REPLACE FUNCTION touch_updated_at() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_cases_touch ON cases;
CREATE TRIGGER trg_cases_touch
BEFORE UPDATE ON cases
FOR EACH ROW EXECUTE FUNCTION touch_updated_at();
