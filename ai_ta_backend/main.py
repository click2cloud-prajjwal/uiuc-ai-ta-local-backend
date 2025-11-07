import warnings
warnings.filterwarnings('ignore')

import asyncio
import os
import time
import logging
from typing import Dict, List

from dotenv import load_dotenv
from flask import Flask, Response, abort, jsonify, request
from flask_cors import CORS
from flask_injector import FlaskInjector, RequestScope
from injector import Binder, SingletonScope
from uuid import uuid4

# --- Core Project Imports ---
from database.blob import BlobStorage      #  new blob storage
from database.sql import SQLDatabase
from service.response_service import ResponseService
from service.retrieval_service import RetrievalService
from rabbitmq.rmqueue import Queue

# --- App Initialization ---
app = Flask(__name__)
CORS(app)
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --------------------------------------------------------------------
# ðŸš€ /INGEST
# --------------------------------------------------------------------
@app.route('/ingest', methods=['POST'])
def ingest() -> Response:
    """
    Uploads a file to Azure Blob Storage and queues it for ingestion.
    """
    from tempfile import gettempdir
    active_queue = Queue()

    try:
        if request.content_type.startswith('multipart/form-data'):
            course_name = request.form.get('course_name')
            readable_filename = request.form.get('readable_filename')
            uploaded_file = request.files.get('file')

            if not uploaded_file:
                return jsonify({"error": "No file uploaded"}), 400

            filename = uploaded_file.filename
            temp_path = os.path.join(gettempdir(), filename)
            uploaded_file.save(temp_path)
            logging.info(f"ðŸ“‚ Saved uploaded file: {temp_path}")

            # --- Upload to Azure Blob ---
            blob = BlobStorage()
            blob_key = f"uploads/{filename}"
            blob.upload_file(temp_path, blob_key)
            logging.info(f"âœ… Uploaded to Azure Blob: {blob_key}")

            # --- Add job to RabbitMQ queue ---
            data = {
                "course_name": course_name,
                "s3_paths": [blob_key],  # keep field name same for compatibility
                "readable_filename": readable_filename,
            }
            job_id = active_queue.addJobToIngestQueue(data)

            return jsonify({
                "outcome": "Queued Ingest task",
                "task_id": job_id,
                "blob_path": blob_key
            }), 200

        elif request.content_type.startswith('application/json'):
            data = request.get_json()
            job_id = active_queue.addJobToIngestQueue(data)
            return jsonify({"outcome": "Queued Ingest task", "task_id": job_id}), 200

        else:
            return jsonify({"error": "Unsupported content type"}), 415

    except Exception as e:
        logging.error(f"âŒ Error in /ingest: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------------------------
# ðŸ’¬ /CHAT
# --------------------------------------------------------------------
@app.route('/chat', methods=['POST'])
def chat(retrieval_service: RetrievalService, response_service: ResponseService) -> Response:
    """
    Handles chat queries using context retrieval + Azure OpenAI response.
    """
    start_time = time.monotonic()
    try:
        data = request.get_json(force=True)
        question: str = data.get('question', '').strip()
        course_name: str = data.get('course_name', '').strip()
        conversation_id: str = data.get('conversation_id', '')
        conversation_history: List[Dict] = data.get('conversation_history', [])

        if not question or not course_name:
            abort(400, description="Missing required parameters: 'question' and 'course_name'")


        logging.info(f"ðŸ’¬ Chat request | Course: {course_name} | Question: {question[:80]}...")

        # --- Step 1: Retrieve Contexts ---
        contexts = asyncio.run(
            retrieval_service.getTopContexts(
                search_query=question,
                course_name=course_name,
                doc_groups=[],
                top_n=5,
                conversation_id=conversation_id
            )
        )

        if not contexts:
            return jsonify({
                "answer": (
                    "I don't have enough information in the course materials "
                    "to answer this question. Try rephrasing or asking about covered topics."
                ),
                "contexts": [],
                "sources_used": 0,
                "model": None
            }), 200

        # --- Step 2: Generate Response ---
        result = response_service.generate_response(
            question=question,
            contexts=contexts,
            course_name=course_name,
            conversation_history=conversation_history,
        )

        # --- Step 3: Save Conversation in SQL ---
        from sqlalchemy import text
        convo_id = conversation_id or str(uuid4())
        model_used = result["model"]

        try:
            with retrieval_service.sqlDb.engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO conversations (id, name, model, project_name, created_at, updated_at)
                        VALUES (:id, :name, :model, :project_name, NOW(), NOW())
                        ON CONFLICT (id) DO NOTHING;
                    """),
                    {"id": convo_id, "name": course_name, "model": model_used, "project_name": course_name},
                )

                conn.execute(
                    text("""
                        INSERT INTO messages (conversation_id, role, content_text, created_at, updated_at, response_time_sec)
                        VALUES (:conversation_id, :role, :content_text, NOW(), NOW(), :response_time_sec);
                    """),
                    {
                        "conversation_id": convo_id,
                        "role": "assistant",
                        "content_text": result["answer"],
                        "response_time_sec": time.monotonic() - start_time,
                    },
                )
        except Exception as e:
            logging.warning(f"Failed to store conversation: {e}")

        response = jsonify({
            "answer": result["answer"],
            "contexts": contexts,
            "sources_used": result["sources_used"],
            "model": result["model"],
            "usage": result.get("usage", {})
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        logging.info(f"âœ… Chat completed in {(time.monotonic() - start_time):.2f} sec")
        return response

    except Exception as e:
        logging.error(f"âŒ Error in /chat: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------------------------
# ðŸ”§ Dependency Injection Configuration
# --------------------------------------------------------------------
def configure(binder: Binder) -> None:
    binder.bind(RetrievalService, to=RetrievalService, scope=RequestScope)
    binder.bind(ResponseService, to=ResponseService, scope=RequestScope)
    binder.bind(SQLDatabase, to=SQLDatabase, scope=SingletonScope)
    binder.bind(BlobStorage, to=BlobStorage, scope=SingletonScope)

FlaskInjector(app=app, modules=[configure])

# --------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0', use_reloader=False, port=int(os.getenv("PORT", 5000)))
