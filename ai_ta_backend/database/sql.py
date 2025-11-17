import logging
import os
from contextlib import contextmanager
from typing import List, Any, Dict

from sqlalchemy import create_engine, select, insert, delete, text, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from sqlalchemy.pool import NullPool

try:
    import rabbitmq.models as models
except ModuleNotFoundError:
    import models

Base = declarative_base()


def orm_to_dict(obj):
    """Convert a SQLAlchemy ORM instance to a dictionary."""
    if obj is None:
        return None
    if hasattr(obj, "__table__"):
        return {col.name: getattr(obj, col.name) for col in obj.__table__.columns}
    return obj


class SQLDatabase:
    """
    Simplified SQL interface for ingestion, response, and chat.
    """

    def __init__(self) -> None:
        db_uri = self._build_db_uri()
        logging.info(f"🧠 Connecting to database: {db_uri}")
        self.engine = create_engine(db_uri, poolclass=NullPool)
        self.Session = sessionmaker(bind=self.engine)
        logging.info("✅ Database connection established.")

    def _build_db_uri(self) -> str:
        if os.getenv("POSTGRES_USERNAME") and os.getenv("POSTGRES_ENDPOINT"):
            return (
                f"postgresql://{os.getenv('POSTGRES_USERNAME')}:"
                f"{os.getenv('POSTGRES_PASSWORD')}@"
                f"{os.getenv('POSTGRES_ENDPOINT')}:"
                f"{os.getenv('POSTGRES_PORT', '5432')}/"
                f"{os.getenv('POSTGRES_DATABASE')}"
            )
        elif os.getenv("SQLITE_DB_NAME"):
            return f"sqlite:///{os.getenv('SQLITE_DB_NAME')}"
        else:
            raise ValueError("❌ Missing required DB environment variables.")

    @contextmanager
    def get_session(self):
        """Context manager for SQLAlchemy sessions."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ==========================================================
    #  📦 Used in Ingest (document management)
    # ==========================================================

    def insert_document(self, document: Dict[str, Any]) -> bool:
        """Insert a single document into the database."""
        try:
            with self.get_session() as session:
                stmt = insert(models.Document).values(document)
                session.execute(stmt)
                logging.info("✅ Document inserted successfully.")
            return True
        except SQLAlchemyError as e:
            logging.error(f"❌ Failed to insert document: {e}")
            return False

    def insert_failed_document(self, document: Dict[str, Any]):
        """Record a failed document ingestion attempt."""
        try:
            with self.get_session() as session:
                stmt = insert(models.FailedDocument).values(document)
                session.execute(stmt)
                logging.warning("⚠️ Failed document recorded.")
        except SQLAlchemyError as e:
            logging.error(f"❌ Failed to record failed document: {e}")

    def delete_document_by_blob_path(self, course_name: str, blob_path: str):
        """Delete document records by blob path."""
        try:
            with self.get_session() as session:
                stmt = (
                    delete(models.Document)
                    .where(models.Document.blob_path == blob_path)
                    .where(models.Document.course_name == course_name)
                )
                session.execute(stmt)
                logging.info(f"🗑️ Deleted document {blob_path} from {course_name}")
        except Exception as e:
            logging.error(f"❌ Failed to delete document by blob_path: {e}")

    def delete_document_by_url(self, course_name: str, url: str):
        """Delete document records by source URL."""
        try:
            with self.get_session() as session:
                stmt = (
                    delete(models.Document)
                    .where(models.Document.url == url)
                    .where(models.Document.course_name == course_name)
                )
                session.execute(stmt)
                logging.info(f"🗑️ Deleted document {url} from {course_name}")
        except Exception as e:
            logging.error(f"❌ Failed to delete document by URL: {e}")

    def delete_document_in_progress(self, job_id: str):
        """Clean up document_in_progress entry after job completion."""
        try:
            with self.get_session() as session:
                stmt = (
                    delete(models.DocumentsInProgress)
                    .where(models.DocumentsInProgress.job_id == job_id)
                )
                session.execute(stmt)
                logging.info(f"🧹 Removed document in progress for job_id={job_id}")
        except Exception as e:
            logging.warning(f"⚠️ Failed to delete in-progress document: {e}")

    def get_like_docs_by_blob_path(self, course_name: str, filename: str):
        """Check for similar documents (used for duplicate detection)."""
        query = (
            select(models.Document)
            .where(models.Document.course_name == course_name)
            .where(models.Document.blob_path.like(f"%{filename}%"))
        )
        with self.get_session() as session:
            result = session.execute(query).scalars().all()
            data = [orm_to_dict(doc) for doc in result]
            return {"data": data}

    def get_like_docs_by_url(self, course_name: str, url: str):
        """Check for documents with similar URLs (for duplicates)."""
        query = (
            select(models.Document)
            .where(models.Document.course_name == course_name)
            .where(models.Document.url.like(f"%{url}%"))
        )
        with self.get_session() as session:
            result = session.execute(query).scalars().all()
            data = [orm_to_dict(doc) for doc in result]
            return {"data": data}

    def insert_doc_group(self, name: str, course_name: str, enabled: bool = True, private: bool = True, doc_count: int = 0):
        """Insert or retrieve a doc_group entry."""
        try:
            with self.get_session() as session:
                existing = (
                    session.query(models.DocGroup)
                    .filter_by(name=name, course_name=course_name)
                    .first()
                )
                if existing:
                    logging.info(f"⚠️ Doc group '{name}' already exists for course '{course_name}'.")
                    return existing.id

                new_group = models.DocGroup(
                    name=name,
                    course_name=course_name,
                    enabled=enabled,
                    private=private,
                    doc_count=doc_count,
                )
                session.add(new_group)
                session.flush()  # get the new ID before commit
                logging.info(f"✅ Created new doc_group '{name}' for course '{course_name}'.")
                return new_group.id
        except Exception as e:
            logging.error(f"❌ Failed to insert doc_group '{name}': {e}")
            return None

    # ==========================================================
    # 💬 Used in Chat / Response (conversations + stats)
    # ==========================================================

    def updateProjectStats(self, project_name: str, model_name: str, is_new_conversation: bool = False):
        """Increment message/conversation counts and model usage."""
        try:
            conversation_increment = 1 if is_new_conversation else 0
            query = text("""
                INSERT INTO project_stats (
                    project_name, total_messages, total_conversations, model_usage_counts, created_at, updated_at
                )
                VALUES (
                    :project_name, 1, :conversation_increment, jsonb_build_object(:model_name, 1), NOW(), NOW()
                )
                ON CONFLICT (project_name)
                DO UPDATE SET
                    total_messages = project_stats.total_messages + 1,
                    total_conversations = project_stats.total_conversations + :conversation_increment,
                    model_usage_counts = 
                        CASE 
                            WHEN project_stats.model_usage_counts ? :model_name THEN
                                jsonb_set(
                                    project_stats.model_usage_counts,
                                    ARRAY[:model_name],
                                    to_jsonb(
                                        (project_stats.model_usage_counts ->> :model_name)::int + 1
                                    )
                                )
                            ELSE 
                                project_stats.model_usage_counts || jsonb_build_object(:model_name, 1)
                        END,
                    updated_at = NOW();
            """)
            with self.get_session() as session:
                session.execute(query, {
                    "project_name": project_name,
                    "model_name": model_name,
                    "conversation_increment": conversation_increment
                })
                logging.info(f"📊 Updated project stats for {project_name} | Model: {model_name}")
        except Exception as e:
            logging.error(f"❌ Failed to update project stats: {e}")

    # ==========================================================
    # 🧠 Basic Helper Query (for chat insertions)
    # ==========================================================

    def insert_conversation_message(self, convo_id: str, course_name: str, model: str, answer: str, response_time_sec: float):
        """Insert conversation and assistant message record."""
        try:
            with self.get_session() as session:
                # Insert conversation if new
                session.execute(
                    text("""
                        INSERT INTO conversations (id, name, model, project_name, created_at, updated_at)
                        VALUES (:id, :name, :model, :project_name, NOW(), NOW())
                        ON CONFLICT (id) DO NOTHING;
                    """),
                    {"id": convo_id, "name": course_name, "model": model, "project_name": course_name},
                )

                # Insert assistant message
                session.execute(
                    text("""
                        INSERT INTO messages (conversation_id, role, content_text, created_at, updated_at, response_time_sec)
                        VALUES (:conversation_id, :role, :content_text, NOW(), NOW(), :response_time_sec);
                    """),
                    {
                        "conversation_id": convo_id,
                        "role": "assistant",
                        "content_text": answer,
                        "response_time_sec": response_time_sec,
                    },
                )
                logging.info(f"💾 Stored message for conversation {convo_id}")
        except Exception as e:
            logging.warning(f"⚠️ Failed to insert conversation message: {e}")
