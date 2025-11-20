import asyncio
import os
import time
import traceback
import logging
from typing import Dict, List, Union
from injector import inject

from langchain.schema import Document
from langchain_community.embeddings import AzureOpenAIEmbeddings
from qdrant_client.http import models

from database.blob import BlobStorage
from database.sql import SQLDatabase
from database.vector import VectorDatabase
from executors.thread_pool_executor import ThreadPoolExecutorAdapter


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

        # Azure OpenAI Embeddings 
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.environ["EMBEDDING_MODEL"],
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            chunk_size=1000,
        )
        logging.info("✅ Using Azure OpenAI embeddings")

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

        if doc_groups is None:
            doc_groups = []

        try:
            start_time = time.monotonic()
            logging.info(f"🔍 Retrieving contexts for course: {course_name}")

            # Compute query embedding (CLEAN)
            user_query_embedding = self._embed_query_and_measure_latency(
                search_query, self.embeddings
            )

            # Vector search
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

        start_time = time.monotonic()
        search_results = []

        try:
            # Combine course filter + conversation filter
            if course_name and course_name.strip():
                allowed_courses = [course_name, "Global"]
            else:
                # User selected nothing -> only search global
                allowed_courses = ["Global"]

            regular_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="course_name",
                        match=models.MatchAny(any=allowed_courses)
                    )
                ]
            )
            if conversation_id:

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
                search_results = self.vdb.vector_search_with_filter(
                    search_query,
                    course_name,
                    doc_groups,
                    user_query_embedding,
                    top_n,
                    [],
                    [],
                    regular_filter,   # USE THE SAME FILTER FROM ABOVE
                )


        except Exception as e:
            logging.error(f"❌ Vector search failed: {e}")

        # Measure Qdrant latency
        self.qdrant_latency_sec = time.monotonic() - start_time

        # Process results
        return self._process_search_results(search_results, course_name)

    # ---------------------------------------------------------------------
    # 🧠 EMBEDDINGS 
    # ---------------------------------------------------------------------
    def _embed_query_and_measure_latency(self, search_query, embedding_client, *_):
        """Embed query WITHOUT Qwen logic, WITHOUT embedding_model."""
        openai_start_time = time.monotonic()

        text_to_embed = search_query  # Direct query text

        user_query_embedding = embedding_client.embed_query(text_to_embed)

        self.openai_embedding_latency = time.monotonic() - openai_start_time
        return user_query_embedding

    # ---------------------------------------------------------------------
    # 🧩 CONVERSATION FILTER
    # ---------------------------------------------------------------------
    def _create_conversation_filter(self, conversation_id: str):
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="conversation_id",
                    match=models.MatchValue(value=conversation_id),
                )
            ]
        )

    # ---------------------------------------------------------------------
    # 🗃️ PROCESS QDRANT RESULTS
    # ---------------------------------------------------------------------
    def _process_search_results(self, search_results, course_name):
        found_docs: list[Document] = []

        for d in search_results or []:
            try:
                metadata = d.payload
                page_content = metadata.get("page_content", "")

                if not page_content.strip():
                    continue

                metadata.pop("page_content", None)

                # Normalize page number field
                if "pagenumber" not in metadata and "pagenumber_or_timestamp" in metadata:
                    metadata["pagenumber"] = metadata["pagenumber_or_timestamp"]

                found_docs.append(Document(page_content=page_content, metadata=metadata))

            except Exception as e:
                logging.error(f"⚠️ Error processing doc in {course_name}: {e}")

        return found_docs

    # ---------------------------------------------------------------------
    # 📦 FORMAT OUTPUT FOR JSON
    # ---------------------------------------------------------------------
    def format_for_json(self, found_docs: List[Document]) -> List[Dict]:
        return [
            {
                "text": doc.page_content,
                "readable_filename": doc.metadata.get("readable_filename"),
                "course_name": doc.metadata.get("course_name"),
                "blob_path": doc.metadata.get("blob_path"),
                "pagenumber": doc.metadata.get("pagenumber"),
                "url": doc.metadata.get("url"),
                "base_url": doc.metadata.get("base_url"),
                "doc_groups": doc.metadata.get("doc_groups"),
            }
            for doc in found_docs
        ]
