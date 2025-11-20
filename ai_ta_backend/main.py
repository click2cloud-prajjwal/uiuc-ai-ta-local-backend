import warnings
warnings.filterwarnings('ignore')
import uuid
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
from database.blob import BlobStorage      # new blob storage
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
# 🚀 /INGEST
# --------------------------------------------------------------------
@app.route('/ingest', methods=['POST'])
def ingest() -> Response:
    """
    Uploads one or multiple files to Azure Blob Storage and queues them for ingestion.
    """
    from tempfile import gettempdir
    active_queue = Queue()

    try:
        if request.content_type.startswith('multipart/form-data'):
            course_name = request.form.get('course_name')
            readable_filename = request.form.get('readable_filename')

            # ✅ Handle one or many files (Postman: key must be 'file')
            uploaded_files = request.files.getlist('file')
            if not uploaded_files:
                return jsonify({"error": "No files uploaded"}), 400

            # ✅ Accept groups from form (supports both keys + comma separated)
            raw_groups = request.form.getlist('doc_groups') or request.form.getlist('groups')
            if len(raw_groups) == 1 and ',' in raw_groups[0]:
                doc_groups = [g.strip() for g in raw_groups[0].split(',') if g.strip()]
            else:
                doc_groups = raw_groups

            # Optional passthroughs (safe defaults)
            force_embeddings = request.form.get('force_embeddings', 'false').lower() == 'true'
            url = request.form.get('url')
            base_url = request.form.get('base_url')

            blob = BlobStorage()
            blob_keys = []

            # ✅ Loop over all uploaded files
            for uploaded_file in uploaded_files:
                filename = uploaded_file.filename
                temp_path = os.path.join(gettempdir(), filename)
                uploaded_file.save(temp_path)
                logging.info(f"📂 Saved uploaded file: {temp_path}")

                # --- Upload each to Azure Blob ---
                blob_key = f"uploads/{uuid.uuid4()}_{filename}"
                blob.upload_file(temp_path, blob_key)
                logging.info(f"✅ Uploaded to Azure Blob: {blob_key}")

                blob_keys.append(blob_key)

                # Clean up temp file
                try:
                    os.remove(temp_path)
                except Exception as cleanup_err:
                    logging.warning(f"⚠️ Could not delete temp file {temp_path}: {cleanup_err}")

            # --- Add one job to RabbitMQ queue with all uploaded files ---
            data = {
                "course_name": course_name,
                "blob_path": blob_keys,  # ✅ list of all files
                "readable_filename": readable_filename,

                # Backward-compatible group forwarding
                "doc_groups": doc_groups,
                "language": doc_groups,

                # Optional flags/fields
                "force_embeddings": force_embeddings,
                "url": url,
                "base_url": base_url,
            }
            job_id = active_queue.addJobToIngestQueue(data)

            return jsonify({
                "outcome": f"Queued {len(blob_keys)} files for ingestion",
                "task_id": job_id,
                "blob_paths": blob_keys
            }), 200

        elif request.content_type.startswith('application/json'):
            data = request.get_json()
            job_id = active_queue.addJobToIngestQueue(data)
            return jsonify({"outcome": "Queued Ingest task", "task_id": job_id}), 200

        else:
            return jsonify({"error": "Unsupported content type"}), 415

    except Exception as e:
        logging.error(f"❌ Error in /ingest: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# # --------------------------------------------------------------------
# /CHAT  — FULL MULTILINGUAL RAG PIPELINE
# --------------------------------------------------------------------
@app.route('/chat', methods=['POST'])
def chat(retrieval_service: RetrievalService, response_service: ResponseService) -> Response:
    start_time = time.monotonic()

    try:
        data = request.get_json(force=True)
        question: str = data.get('question', '').strip()
        course_name: str = data.get('course_name', '').strip()
        conversation_id: str = data.get('conversation_id', '')
        conversation_history: List[Dict] = data.get('conversation_history', [])

        # Accept both "doc_groups" or "groups"
        doc_groups = data.get('doc_groups') or data.get('groups') or []
        if isinstance(doc_groups, str) and doc_groups.strip():
            doc_groups = [doc_groups]
        elif not isinstance(doc_groups, list):
            doc_groups = []

        if not question :
            abort(400, description="Missing required parameters: 'question' ")

        logging.info(f"Chat request | Course: {course_name} | Groups: {doc_groups} | Question: {question[:80]}...")

        # === Multilingual Step 1: Detect language of user query ===
        original_lang = response_service.detect_language(question)
        logging.info(f"Detected user language: {original_lang}")

        # === Multilingual Step 2: Determine document language from doc_groups ===
        # If no doc_groups provided, default to user language
        if len(doc_groups) > 0:
            target_doc_lang = doc_groups[0].lower()
        else:
            target_doc_lang = original_lang

        logging.info(f"Document language for retrieval: {target_doc_lang}")

        # === Multilingual Step 3: Translate the query BEFORE retrieval ===
        translated_query = question
        if original_lang != target_doc_lang:
            logging.info(f"Translating query {original_lang} -> {target_doc_lang} before retrieval")
            translated_query = response_service.translate(question, target_doc_lang)

        # === Step 4: Retrieve contexts ===
        contexts = asyncio.run(
            retrieval_service.getTopContexts(
                search_query=translated_query,
                course_name=course_name,
                doc_groups=doc_groups,
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

        # === Step 5: Generate the final answer (ResponseService will translate back) ===
        result = response_service.generate_response(
            question=question,  # original question
            contexts=contexts,
            course_name=course_name,
            conversation_history=conversation_history,
        )

        # === Step 6: Store conversation ===
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
        logging.info(f"Chat completed in {(time.monotonic() - start_time):.2f} sec")

        return response

    except Exception as e:
        logging.error(f"Error in /chat: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500



# --------------------------------------------------------------------
# 🔍 /getTopContexts
# --------------------------------------------------------------------
@app.route('/getTopContexts', methods=['POST'])
def getTopContexts(service: RetrievalService) -> Response:
    """
    Get most relevant contexts for a given search query.
    """
    start_time = time.monotonic()
    data = request.get_json()
    search_query: str = data.get('search_query', '')
    course_name: str = data.get('course_name', '')
    doc_groups: List[str] = data.get('doc_groups', [])
    top_n: int = data.get('top_n', 100)
    conversation_id: str = data.get('conversation_id', '')

    if search_query == '' or course_name == '':
        abort(
            400,
            description=(
                f"Missing one or more required parameters: "
                f"'search_query' and 'course_name' must be provided. "
                f"Search query: `{search_query}`, Course name: `{course_name}`"
            )
        )

    found_documents = asyncio.run(service.getTopContexts(search_query, course_name, doc_groups, top_n, conversation_id))
    response = jsonify(found_documents)
    response.headers.add('Access-Control-Allow-Origin', '*')
    print(f"⏰ Runtime of getTopContexts in main.py: {(time.monotonic() - start_time):.2f} seconds")
    return response


# --------------------------------------------------------------------
# ⚙️ Dependency Injection Configuration
# --------------------------------------------------------------------
def configure(binder: Binder) -> None:
    binder.bind(RetrievalService, to=RetrievalService, scope=RequestScope)
    binder.bind(ResponseService, to=ResponseService, scope=RequestScope)
    binder.bind(SQLDatabase, to=SQLDatabase, scope=SingletonScope)
    binder.bind(BlobStorage, to=BlobStorage, scope=SingletonScope)


FlaskInjector(app=app, modules=[configure])

# --------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', use_reloader=False, port=int(os.getenv("PORT", 5000)))
