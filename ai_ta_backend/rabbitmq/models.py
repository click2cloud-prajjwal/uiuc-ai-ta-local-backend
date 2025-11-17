"""
Minimal models for ingestion and chat operations.
"""

from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, ForeignKey, JSON, Text, Float
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from uuid import uuid4

try:
    from rabbitmq.extensions import db
except ModuleNotFoundError:
    from extensions import db


# ---------------------------------------------------------------------
# 🧱 Base model
# ---------------------------------------------------------------------
class Base(db.Model):
    __abstract__ = True


# ---------------------------------------------------------------------
# 📄 Documents (used for ingestion + retrieval)
# ---------------------------------------------------------------------
class Document(Base):
    __tablename__ = 'documents'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=func.now())
    blob_path = Column(Text)  # compatible with Blob key names
    readable_filename = Column(Text)
    course_name = Column(Text)
    url = Column(Text)
    contexts = Column(JSON, default=lambda: [])
    base_url = Column(Text)

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at,
            "blob_path": self.blob_path,
            "readable_filename": self.readable_filename,
            "course_name": self.course_name,
            "url": self.url,
            "contexts": self.contexts,
            "base_url": self.base_url,
        }


# ---------------------------------------------------------------------
# ⚙️ Track In-Progress / Failed Ingest Jobs
# ---------------------------------------------------------------------
class DocumentsInProgress(Base):
    __tablename__ = 'documents_in_progress'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=func.now())
    blob_path = Column(Text)
    readable_filename = Column(Text)
    course_name = Column(Text)
    url = Column(Text)
    contexts = Column(JSON, default=lambda: [])
    base_url = Column(Text)
    doc_groups = Column(Text)
    error = Column(Text)
    beam_task_id = Column(Text, default=lambda: str(uuid4()))

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at,
            "blob_path": self.blob_path,
            "readable_filename": self.readable_filename,
            "course_name": self.course_name,
            "url": self.url,
            "contexts": self.contexts,
            "base_url": self.base_url,
            "doc_groups": self.doc_groups,
            "error": self.error,
            "beam_task_id": self.beam_task_id,
        }


class DocumentsFailed(Base):
    __tablename__ = 'documents_failed'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=func.now())
    blob_path = Column(Text)
    readable_filename = Column(Text)
    course_name = Column(Text)
    url = Column(Text)
    contexts = Column(JSON, default=lambda: [])
    base_url = Column(Text)
    doc_groups = Column(Text)
    error = Column(Text)

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at,
            "blob_path": self.blob_path,
            "readable_filename": self.readable_filename,
            "course_name": self.course_name,
            "url": self.url,
            "contexts": self.contexts,
            "base_url": self.base_url,
            "doc_groups": self.doc_groups,
            "error": self.error,
        }


# ---------------------------------------------------------------------
# 💬 Chat: Conversations and Messages
# ---------------------------------------------------------------------
class Conversations(Base):
    __tablename__ = 'conversations'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(Text)
    model = Column(Text)
    prompt = Column(Text)
    temperature = Column(Float)
    user_email = Column(Text)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    project_name = Column(Text)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "model": self.model,
            "prompt": self.prompt,
            "temperature": self.temperature,
            "user_email": self.user_email,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "project_name": self.project_name,
        }


class Messages(Base):
    __tablename__ = 'messages'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey('conversations.id', ondelete='CASCADE'))
    role = Column(Text)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    content_text = Column(Text)
    response_time_sec = Column(BigInteger)
    contexts = Column(JSON)
    tools = Column(JSON)
    latest_system_message = Column(Text)
    final_prompt_engineered_message = Column(Text)

    def to_dict(self):
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "content_text": self.content_text,
            "response_time_sec": self.response_time_sec,
            "contexts": self.contexts,
            "tools": self.tools,
            "latest_system_message": self.latest_system_message,
            "final_prompt_engineered_message": self.final_prompt_engineered_message,
        }


# ---------------------------------------------------------------------
# 📊 Project Stats (for model usage tracking)
# ---------------------------------------------------------------------
class ProjectStats(Base):
    __tablename__ = 'project_stats'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    project_name = Column(Text)
    total_messages = Column(BigInteger, default=0)
    total_conversations = Column(BigInteger, default=0)
    unique_users = Column(BigInteger, default=0)
    model_usage_counts = Column(JSON, default=lambda: {})
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    def to_dict(self):
        return {
            "id": self.id,
            "project_name": self.project_name,
            "total_messages": self.total_messages,
            "total_conversations": self.total_conversations,
            "unique_users": self.unique_users,
            "model_usage_counts": self.model_usage_counts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
# ---------------------------------------------------------------------
# 🗂️ Doc Groups (stores document group metadata)
# ---------------------------------------------------------------------
class DocGroup(Base):
    __tablename__ = 'doc_groups'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False)
    course_name = Column(Text, nullable=False)
    created_at = Column(DateTime, default=func.now())
    enabled = Column(Boolean, default=True)
    private = Column(Boolean, default=True)
    doc_count = Column(BigInteger, default=0)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "course_name": self.course_name,
            "created_at": self.created_at,
            "enabled": self.enabled,
            "private": self.private,
            "doc_count": self.doc_count,
        }
