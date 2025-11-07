"""
Complete Database Migration - All Tables from models.py
Creates ALL required tables for the AI-TA backend
"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

print("=" * 70)
print("🗄️  COMPLETE DATABASE MIGRATION")
print("=" * 70)

# Get database credentials
POSTGRES_USERNAME = os.getenv('POSTGRES_USERNAME')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_ENDPOINT = os.getenv('POSTGRES_ENDPOINT')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_DATABASE = os.getenv('POSTGRES_DATABASE')

# Build connection string
db_uri = f"postgresql://{POSTGRES_USERNAME}:{POSTGRES_PASSWORD}@{POSTGRES_ENDPOINT}:{POSTGRES_PORT}/{POSTGRES_DATABASE}"

print(f"\n📋 Database: {POSTGRES_ENDPOINT}:{POSTGRES_PORT}/{POSTGRES_DATABASE}")

# Complete SQL with ALL tables from models.py
complete_migration_sql = """
-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Documents table (main documents)
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    s3_path TEXT,
    readable_filename TEXT,
    course_name TEXT,
    url TEXT,
    contexts JSON DEFAULT '[{"text": "", "timestamp": "", "embedding": "", "pagenumber": ""}]',
    base_url TEXT
);

CREATE INDEX IF NOT EXISTS documents_course_name_idx ON documents USING HASH (course_name);
CREATE INDEX IF NOT EXISTS documents_created_at_idx ON documents USING BTREE (created_at);
CREATE INDEX IF NOT EXISTS idx_doc_s3_path ON documents USING BTREE (s3_path);

-- 2. Doc Groups table
CREATE TABLE IF NOT EXISTS doc_groups (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    course_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    enabled BOOLEAN DEFAULT TRUE,
    private BOOLEAN DEFAULT TRUE,
    doc_count BIGINT
);

CREATE INDEX IF NOT EXISTS doc_groups_enabled_course_name_idx ON doc_groups USING BTREE (enabled, course_name);

-- 3. Documents-DocGroups junction table
CREATE TABLE IF NOT EXISTS documents_doc_groups (
    document_id BIGINT,
    doc_group_id BIGINT REFERENCES doc_groups(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (document_id, doc_group_id)
);

CREATE INDEX IF NOT EXISTS documents_doc_groups_doc_group_id_idx ON documents_doc_groups USING BTREE (doc_group_id);
CREATE INDEX IF NOT EXISTS documents_doc_groups_document_id_idx ON documents_doc_groups USING BTREE (document_id);

-- 4. Documents In Progress table
CREATE TABLE IF NOT EXISTS documents_in_progress (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    s3_path TEXT,
    readable_filename TEXT,
    course_name TEXT,
    url TEXT,
    contexts JSON DEFAULT '[{"text": "", "timestamp": "", "embedding": "", "pagenumber": ""}]',
    base_url TEXT,
    doc_groups TEXT,
    error TEXT,
    beam_task_id TEXT DEFAULT gen_random_uuid()::TEXT
);

-- 5. Documents Failed table
CREATE TABLE IF NOT EXISTS documents_failed (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    s3_path TEXT,
    readable_filename TEXT,
    course_name TEXT,
    url TEXT,
    contexts JSON DEFAULT '[{"text": "", "timestamp": "", "embedding": "", "pagenumber": ""}]',
    base_url TEXT,
    doc_groups TEXT,
    error TEXT
);

-- 6. Projects table
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

CREATE UNIQUE INDEX IF NOT EXISTS projects_course_name_key ON projects USING BTREE (course_name);

-- 7. Project Stats table
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

-- 8. N8N Workflows table
CREATE TABLE IF NOT EXISTS n8n_workflows (
    latest_workflow_id BIGSERIAL PRIMARY KEY,
    is_locked BOOLEAN NOT NULL
);

-- 9. LLM Conversation Monitor table
CREATE TABLE IF NOT EXISTS "llm-convo-monitor" (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    convo JSON,
    convo_id TEXT UNIQUE,
    course_name TEXT,
    user_email TEXT
);

CREATE INDEX IF NOT EXISTS llm_convo_monitor_course_name_idx ON "llm-convo-monitor" USING HASH (course_name);

-- 10. Conversations table
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

CREATE INDEX IF NOT EXISTS idx_user_email_updated_at ON conversations USING BTREE (user_email, updated_at);

-- 11. Messages table
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

-- 12. Pre-Authorized API Keys table
CREATE TABLE IF NOT EXISTS pre_authorized_api_keys (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    emails JSONB,
    providerBodyNoModels JSON,
    providerName TEXT,
    notes TEXT
);
"""

try:
    engine = create_engine(db_uri)
    
    print("\n🔌 Connecting to database...")
    with engine.connect() as conn:
        print("\n📦 Creating all tables...")
        conn.execute(text(complete_migration_sql))
        conn.commit()
        
        print("\n✅ Verifying tables...")
        result = conn.execute(text("""
            SELECT 
                table_name,
                (SELECT COUNT(*) FROM information_schema.columns 
                 WHERE table_name = t.table_name AND table_schema = 'public') as column_count
            FROM information_schema.tables t
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """))
        
        tables = list(result)
        print(f"\n   Created {len(tables)} table(s):")
        for table_name, col_count in tables:
            print(f"   ✅ {table_name:35} ({col_count} columns)")
    
    print("\n" + "=" * 70)
    print("✅ COMPLETE MIGRATION SUCCESSFUL!")
    print("=" * 70)
    print("\n📊 Database now contains:")
    print("   • Document management (documents, documents_in_progress, documents_failed)")
    print("   • Document organization (doc_groups, documents_doc_groups)")
    print("   • Project management (projects, project_stats)")
    print("   • Conversation tracking (conversations, messages, llm-convo-monitor)")
    print("   • API key management (pre_authorized_api_keys)")
    print("   • Workflow management (n8n_workflows)")
    print("\n🎯 Your backend is now fully ready!")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\n💡 Troubleshooting:")
    print("   1. Check PostgreSQL is running: docker ps | grep postgres")
    print("   2. Verify .env database credentials")
    import traceback
    traceback.print_exc()