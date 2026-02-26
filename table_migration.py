"""
Complete Database Migration - All Tables from models.py
Creates ALL required tables and functions for the AI-TA backend (with blob_path instead of s3_path)
"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

print("=" * 70)
print("COMPLETE DATABASE MIGRATION (with stored functions + constraints)")
print("=" * 70)

# Get database credentials
POSTGRES_USERNAME = os.getenv('POSTGRES_USERNAME')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_ENDPOINT = os.getenv('POSTGRES_ENDPOINT')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_DATABASE = os.getenv('POSTGRES_DATABASE')

# Build connection string
db_uri = f"postgresql://{POSTGRES_USERNAME}:{POSTGRES_PASSWORD}@{POSTGRES_ENDPOINT}:{POSTGRES_PORT}/{POSTGRES_DATABASE}"

print(f"\nDatabase: {POSTGRES_ENDPOINT}:{POSTGRES_PORT}/{POSTGRES_DATABASE}")

# ============================================================
# UPDATED MIGRATION SQL (blob_path everywhere)
# ============================================================
complete_migration_sql = """
-- ============================================================
-- EXTENSIONS
-- ============================================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================
-- 1. DOCUMENTS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    blob_path TEXT,
    readable_filename TEXT,
    course_name TEXT,
    url TEXT,
    contexts JSON DEFAULT '[{"text": "", "timestamp": "", "embedding": "", "pagenumber": ""}]',
    base_url TEXT
);
CREATE INDEX IF NOT EXISTS documents_course_name_idx ON documents USING HASH (course_name);
CREATE INDEX IF NOT EXISTS idx_doc_blob_path ON documents USING BTREE (blob_path);

-- ============================================================
-- 2. DOC GROUPS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS doc_groups (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    course_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    enabled BOOLEAN DEFAULT TRUE,
    private BOOLEAN DEFAULT TRUE,
    doc_count BIGINT
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'unique_docgroup_name_per_course'
    ) THEN
        ALTER TABLE doc_groups
        ADD CONSTRAINT unique_docgroup_name_per_course UNIQUE (name, course_name);
    END IF;
END
$$;

-- ============================================================
-- 3. DOCUMENTS-DOCGROUPS LINK TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS documents_doc_groups (
    document_id BIGINT REFERENCES documents(id) ON DELETE CASCADE,
    doc_group_id BIGINT REFERENCES doc_groups(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (document_id, doc_group_id)
);
CREATE INDEX IF NOT EXISTS documents_doc_groups_doc_group_id_idx ON documents_doc_groups USING BTREE (doc_group_id);

-- ============================================================
-- 4. DOCUMENTS IN PROGRESS
-- ============================================================
CREATE TABLE IF NOT EXISTS documents_in_progress (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    blob_path TEXT,
    readable_filename TEXT,
    course_name TEXT,
    url TEXT,
    contexts JSON DEFAULT '[{"text": "", "timestamp": "", "embedding": "", "pagenumber": ""}]',
    base_url TEXT,
    doc_groups TEXT,
    error TEXT,
    beam_task_id TEXT DEFAULT gen_random_uuid()::TEXT
);

-- ============================================================
-- 5. DOCUMENTS FAILED
-- ============================================================
CREATE TABLE IF NOT EXISTS documents_failed (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    blob_path TEXT,
    readable_filename TEXT,
    course_name TEXT,
    url TEXT,
    contexts JSON DEFAULT '[{"text": "", "timestamp": "", "embedding": "", "pagenumber": ""}]',
    base_url TEXT,
    doc_groups TEXT,
    error TEXT
);

-- ============================================================
-- 6. PROJECTS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS projects (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    course_name TEXT,
    doc_map_id TEXT,
    convo_map_id TEXT,
    n8n_api_key TEXT,
    last_uploaded_doc_id BIGINT,
    last_uploaded_convo_id BIGINT,
    subscribed BIGINT REFERENCES doc_groups(id) ON UPDATE CASCADE ON DELETE SET NULL,
    description TEXT,
    metadata_schema JSON
);
CREATE UNIQUE INDEX IF NOT EXISTS projects_course_name_key ON projects(course_name);

-- ============================================================
-- 7. PROJECT STATS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS project_stats (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT,
    project_name TEXT,
    total_messages BIGINT,
    total_conversations BIGINT,
    unique_users BIGINT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    model_usage_counts JSON
);

-- ============================================================
-- 8. N8N WORKFLOWS
-- ============================================================
CREATE TABLE IF NOT EXISTS n8n_workflows (
    latest_workflow_id BIGSERIAL PRIMARY KEY,
    is_locked BOOLEAN NOT NULL
);

-- ============================================================
-- 9. LLM CONVO MONITOR
-- ============================================================
CREATE TABLE IF NOT EXISTS "llm-convo-monitor" (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    convo JSON,
    convo_id TEXT UNIQUE,
    course_name TEXT,
    user_email TEXT
);

-- ============================================================
-- 10. CONVERSATIONS & MESSAGES
-- ============================================================
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR,
    model VARCHAR,
    prompt TEXT,
    temperature FLOAT,
    user_email VARCHAR,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    project_name TEXT,
    folder_id UUID
);

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    contexts JSON,
    tools JSON,
    latest_system_message TEXT,
    final_prompt_engineered_message TEXT,
    response_time_sec BIGINT,
    content_text TEXT,
    updated_at TIMESTAMP DEFAULT NOW(),
    content_image_url TEXT,
    image_description TEXT
);

-- ============================================================
-- 11. PRE-AUTHORIZED API KEYS
-- ============================================================
CREATE TABLE IF NOT EXISTS pre_authorized_api_keys (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    emails JSONB,
    providerBodyNoModels JSON,
    providerName TEXT,
    notes TEXT
);

-- ============================================================
-- STORED PROCEDURES (for ingestion)
-- ============================================================
CREATE OR REPLACE FUNCTION add_document_to_group(
    p_course_name TEXT,
    p_blob_path TEXT,
    p_url TEXT,
    p_readable_filename TEXT,
    p_doc_groups TEXT[]
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO doc_groups (name, course_name, enabled, private, doc_count)
    SELECT unnest(p_doc_groups), p_course_name, TRUE, FALSE, 0
    ON CONFLICT (name, course_name) DO NOTHING;

    INSERT INTO documents_doc_groups (document_id, doc_group_id)
    SELECT d.id, g.id
    FROM documents d
    JOIN doc_groups g ON g.name = ANY(p_doc_groups) AND g.course_name = p_course_name
    WHERE d.blob_path = p_blob_path OR d.url = p_url;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION add_document_to_group_url(
    p_course_name TEXT,
    p_blob_path TEXT,
    p_url TEXT,
    p_readable_filename TEXT,
    p_doc_groups TEXT[]
)
RETURNS VOID AS $$
BEGIN
    PERFORM add_document_to_group(p_course_name, p_blob_path, p_url, p_readable_filename, p_doc_groups);
END;
$$ LANGUAGE plpgsql;
"""

# ============================================================
# EXECUTE MIGRATION
# ============================================================
try:
    engine = create_engine(db_uri)
    print("\nConnecting to database...")
    with engine.connect() as conn:
        print("\nCreating tables and stored functions...")
        conn.execute(text(complete_migration_sql))
        conn.commit()

        print("\nVerifying tables...")
        result = conn.execute(text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """))
        tables = [row[0] for row in result]
        print(f"\nCreated {len(tables)} tables:")
        for t in tables:
            print(f"   {t}")

    print("\n" + "=" * 70)
    print("COMPLETE MIGRATION SUCCESSFUL")
    print("=" * 70)
    print("\nIncludes:")
    print("   • blob_path renamed everywhere (formerly s3_path)")
    print("   • UNIQUE constraint on doc_groups(name, course_name) handled safely")
    print("   • Stored procedures add_document_to_group & add_document_to_group_url")
    print("   • All core document and conversation tables ready")
    print("=" * 70)

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
