import asyncio
import os
import time
import traceback
import logging
from typing import Dict, List, Union
from injector import inject

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.schema import Document
from qdrant_client.http import models

from database.blob import BlobStorage  # ✅ replaced AWSStorage
from database.sql import SQLDatabase
from database.vector import VectorDatabase
from executors.thread_pool_executor import ThreadPoolExecutorAdapter



DEFAULT_QWEN_QUERY_INSTRUCTION = (
    "Given a user search query, retrieve the most relevant passages "
    "from the knowledge base stored in Qdrant to answer the query accurately. "
    "Prioritize authoritative course materials and official documents."
)


class RetrievalService:
    """
    Retrieval service handles vector search + context fetching from Qdrant.
    Used by /chat route to get top contexts for a given user query.
    """

    @inject
    def __init__(
        self,
        vdb: VectorDatabase,
        sqlDb: SQLDatabase,
        blob: BlobStorage,
        thread_pool_executor: ThreadPoolExecutorAdapter,
    ):
        self.vdb = vdb
        self.sqlDb = sqlDb
        self.blob = blob
        self.thread_pool_executor = thread_pool_executor

        # Embedding configuration
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

        if os.getenv("OPENAI_API_TYPE") == "azure":
            from langchain_community.embeddings import AzureOpenAIEmbeddings

            self.embeddings = AzureOpenAIEmbeddings(
                azure_deployment=os.environ["AZURE_OPENAI_ENGINE"],
                openai_api_key=os.environ["AZURE_OPENAI_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                openai_api_version=os.environ["OPENAI_API_VERSION"],
                chunk_size=1000,
            )
            logging.info("✅ Using Azure OpenAI embeddings")
        else:
            self.embeddings = OllamaEmbeddings(
                base_url=os.environ.get("OLLAMA_SERVER_URL", "http://localhost:11434"),
                model="nomic-embed-text:v1.5",
            )
            logging.info("✅ Using Ollama embeddings (local)")

        # Optional override
        self.qwen_query_instruction = os.getenv(
            "QWEN_QUERY_INSTRUCTION", DEFAULT_QWEN_QUERY_INSTRUCTION
        )

    # ---------------------------------------------------------------------
    # 🔍 MAIN RETRIEVAL FUNCTION
    # ---------------------------------------------------------------------
    async def getTopContexts(
        self,
        search_query: str,
        course_name: str,
        doc_groups: List[str] | None = None,
        top_n: int = 5,
        conversation_id: str = "",
    ) -> Union[List[Dict], str]:
        """
        Retrieve top relevant document contexts for a given query.
        """
        if doc_groups is None:
            doc_groups = []
        try:
            start_time = time.monotonic()
            logging.info(f"🔍 Retrieving contexts for course: {course_name}")

            # Compute query embedding
            user_query_embedding = self._embed_query_and_measure_latency(
                search_query, self.embeddings, self.qwen_query_instruction
            )

            # Perform vector search
            found_docs = self.vector_search(
                search_query=search_query,
                course_name=course_name,
                doc_groups=doc_groups,
                user_query_embedding=user_query_embedding,
                top_n=top_n,
                conversation_id=conversation_id,
            )

            if not found_docs:
                logging.warning(f"⚠️ No contexts found for {course_name}")
                return []

            logging.info(
                f"✅ Retrieved {len(found_docs)} docs in {(time.monotonic() - start_time):.2f}s"
            )
            return self.format_for_json(found_docs)

        except Exception as e:
            err = (
                f"ERROR in getTopContexts for {course_name}: {e}\n"
                f"{traceback.format_exc()}"
            )
            logging.error(err)
            return err

    # ---------------------------------------------------------------------
    # ⚙️ VECTOR SEARCH WRAPPER
    # ---------------------------------------------------------------------
    def vector_search(
        self,
        search_query: str,
        course_name: str,
        doc_groups: List[str],
        user_query_embedding,
        top_n: int = 5,
        conversation_id: str = "",
    ):
        """
        Perform vector search on Qdrant for a given query and course.
        Includes conversation-based filtering if applicable.
        """

        start_time = time.monotonic()
        search_results = []

        try:
            if conversation_id:
                # Combine filters: course and conversation documents
                regular_filter = self.vdb._create_search_filter(
                    course_name, doc_groups, [], []
                )
                convo_filter = self._create_conversation_filter(conversation_id)
                combined_filter = models.Filter(should=[regular_filter, convo_filter])

                search_results = self.vdb.vector_search_with_filter(
                    search_query,
                    course_name,
                    doc_groups,
                    user_query_embedding,
                    top_n,
                    [],
                    [],
                    combined_filter,
                )
            else:
                search_results = self.vdb.vector_search(
                    search_query,
                    course_name,
                    doc_groups,
                    user_query_embedding,
                    top_n,
                    [],
                    [],
                )

        except Exception as e:
            logging.error(f"❌ Vector search failed: {e}")

        self.qdrant_latency_sec = time.monotonic() - start_time

        found_docs = self._process_search_results(search_results, course_name)
        return found_docs

    # ---------------------------------------------------------------------
    # 🧠 EMBEDDING
    # ---------------------------------------------------------------------
    def _embed_query_and_measure_latency(
        self, search_query, embedding_client, query_instruction: str | None = None
    ):
        openai_start_time = time.monotonic()
        text_to_embed = search_query

        try:
            model_name = getattr(embedding_client, "model", self.embedding_model)
        except Exception:
            model_name = self.embedding_model

        if (
            query_instruction
            and isinstance(embedding_client, OpenAIEmbeddings)
            and "qwen" in str(model_name).lower()
        ):
            text_to_embed = f"Instruct: {query_instruction}\nQuery:{search_query}"

        user_query_embedding = embedding_client.embed_query(text_to_embed)
        self.openai_embedding_latency = time.monotonic() - openai_start_time
        return user_query_embedding

    # ---------------------------------------------------------------------
    # 🧩 FILTERING & CONVERSATION SUPPORT
    # ---------------------------------------------------------------------
    def _create_conversation_filter(self, conversation_id: str):
        """Create Qdrant filter for conversation-specific content."""
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="conversation_id",
                    match=models.MatchValue(value=conversation_id),
                )
            ]
        )

    # ---------------------------------------------------------------------
    # 🗃️ PROCESS & FORMAT RESULTS
    # ---------------------------------------------------------------------
    def _process_search_results(self, search_results, course_name):
        """Process vector search results into structured LangChain docs."""
        found_docs: list[Document] = []
        for d in search_results or []:
            try:
                metadata = d.payload
                page_content = metadata.get("page_content", "")
                if not page_content.strip():
                    continue
                metadata.pop("page_content", None)

                # Compatibility cleanup
                if "pagenumber" not in metadata and "pagenumber_or_timestamp" in metadata:
                    metadata["pagenumber"] = metadata["pagenumber_or_timestamp"]

                found_docs.append(Document(page_content=page_content, metadata=metadata))
            except Exception as e:
                logging.error(f"⚠️ Error processing doc in {course_name}: {e}")
        return found_docs

    def format_for_json(self, found_docs: List[Document]) -> List[Dict]:
        """Convert retrieved documents into JSON-ready format."""
        return [
            {
                "text": doc.page_content,
                "readable_filename": doc.metadata.get("readable_filename"),
                "course_name": doc.metadata.get("course_name"),
                "s3_path": doc.metadata.get("s3_path"),
                "pagenumber": doc.metadata.get("pagenumber"),
                "url": doc.metadata.get("url"),
                "base_url": doc.metadata.get("base_url"),
                "doc_groups": doc.metadata.get("doc_groups"),
            }
            for doc in found_docs
        ]
    # ---------------------------------------------------------------------