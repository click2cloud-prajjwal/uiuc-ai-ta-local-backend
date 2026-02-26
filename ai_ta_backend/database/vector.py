import os
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from injector import inject
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import FieldCondition, MatchAny, MatchValue


class VectorDatabase:
    """
    Handles all vector database operations using Qdrant.
    """

    @inject
    def __init__(self):
        """
        Initialize Qdrant (local or cloud) and Azure OpenAI embeddings.
        """
        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.environ.get("QDRANT_API_KEY", "")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))

        # Initialize Qdrant client (local or cloud)
        if "localhost" in qdrant_url or "127.0.0.1" in qdrant_url:
            print("🧠 Using LOCAL Qdrant instance")
            self.qdrant_client = QdrantClient(
                url=qdrant_url,
                port=qdrant_port,
                https=False,
                api_key=None,
                timeout=20
            )
        else:
            print("☁️ Using CLOUD Qdrant instance")
            self.qdrant_client = QdrantClient(
                url=qdrant_url,
                port=qdrant_port,
                https=True,
                api_key=qdrant_api_key,
                timeout=20
            )

        # Optional CropWizard Qdrant connection
        try:
            self.cropwizard_qdrant_client = QdrantClient(
                url="https://cropwizard-qdrant.ncsa.ai",
                port=443,
                https=True,
                api_key=os.environ.get("QDRANT_API_KEY")
            )
        except Exception as e:
            print(f"⚠️ CropWizard Qdrant unavailable: {e}")
            self.cropwizard_qdrant_client = None

        # Azure OpenAI embeddings
        self.vectorstore = Qdrant(
            client=self.qdrant_client,
            collection_name=os.environ["QDRANT_COLLECTION_NAME"],
            embeddings=AzureOpenAIEmbeddings(
                azure_deployment=os.environ["EMBEDDING_MODEL"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                chunk_size=1
            )
        )

    # ============================================================
    # Core Vector Search
    # ============================================================

    def vector_search(self, search_query, course_name, doc_groups: List[str], user_query_embedding, top_n,
                      disabled_doc_groups: List[str], public_doc_groups: List[dict]):
        """
        Search in main Qdrant collection.
        """
        search_results = self.qdrant_client.search(
            collection_name=os.environ["QDRANT_COLLECTION_NAME"],
            query_filter=self._create_search_filter(course_name, doc_groups, disabled_doc_groups, public_doc_groups),
            with_vectors=False,
            query_vector=user_query_embedding,
            limit=top_n,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(rescore=False)
            )
        )
        return search_results

    def cropwizard_vector_search(self, search_query, course_name, doc_groups: List[str],
                                 user_query_embedding, top_n, disabled_doc_groups: List[str],
                                 public_doc_groups: List[dict]):
        """
        Search the CropWizard Qdrant (if available).
        """
        if not self.cropwizard_qdrant_client:
            print("⚠️ CropWizard Qdrant not initialized.")
            return []

        search_results = self.cropwizard_qdrant_client.search(
            collection_name="cropwizard",
            query_filter=self._create_search_filter(course_name, doc_groups, disabled_doc_groups, public_doc_groups),
            with_vectors=False,
            query_vector=user_query_embedding,
            limit=top_n
        )
        return search_results

    # ============================================================
    # Filters and Helpers
    # ============================================================

    def _create_search_filter(self, course_name: str, doc_groups: List[str],
                              admin_disabled_doc_groups: List[str], public_doc_groups: List[dict]) -> models.Filter:
        """
        Build search filter for Qdrant based on course and doc groups.
        """
        must_conditions = []
        should_conditions = []
        must_not_conditions = []

        # Exclude disabled doc groups
        if admin_disabled_doc_groups:
            must_not_conditions.append(
                FieldCondition(key="doc_groups", match=MatchAny(any=admin_disabled_doc_groups))
            )

        # Only include chunks with no conversation_id (normal docs)
        must_conditions.append(
            models.IsEmptyCondition(is_empty={"key": "conversation_id"})
        )

        # Handle public doc groups
        if public_doc_groups:
            for public_doc_group in public_doc_groups:
                if public_doc_group.get("enabled", False):
                    combined_condition = models.Filter(must=[
                        FieldCondition(key="course_name", match=MatchValue(value=public_doc_group["course_name"])),
                        FieldCondition(key="doc_groups", match=MatchAny(any=[public_doc_group["name"]]))
                    ])
                    should_conditions.append(combined_condition)

        # Handle own course docs
        own_course_condition = models.Filter(must=[
            FieldCondition(key="course_name", match=MatchValue(value=course_name))
        ])
        if doc_groups and "All Documents" not in doc_groups:
            own_course_condition.must.append(
                FieldCondition(key="doc_groups", match=MatchAny(any=doc_groups))
            )
        should_conditions.append(own_course_condition)

        vector_search_filter = models.Filter(
            must=must_conditions,
            should=should_conditions,
            must_not=must_not_conditions
        )

        print(f"Vector search filter: {vector_search_filter}")
        return vector_search_filter

    def _create_conversation_search_filter(self, conversation_id: str) -> models.Filter:
        """
        Filter for conversation-specific context.
        """
        return models.Filter(
            must=[
                FieldCondition(
                    key="conversation_id",
                    match=MatchValue(value=conversation_id)
                )
            ]
        )

    def _combine_filters(self, search_filter: models.Filter, conversation_filter: models.Filter = None) -> models.Filter:
        """
        Combine search filter with conversation filter.
        """
        combined_conditions = []
        if search_filter.must:
            combined_conditions.extend(search_filter.must)
        if conversation_filter and conversation_filter.must:
            combined_conditions.extend(conversation_filter.must)

        combined_must_not = []
        if search_filter.must_not:
            combined_must_not.extend(search_filter.must_not)
        if conversation_filter and conversation_filter.must_not:
            combined_must_not.extend(conversation_filter.must_not)

        return models.Filter(must=combined_conditions, must_not=combined_must_not)

    # ============================================================
    # Maintenance Functions
    # ============================================================

    def delete_data(self, collection_name: str, key: str, value: str):
        """
        Delete data from the main Qdrant collection.
        """
        return self.qdrant_client.delete(
            collection_name=collection_name,
            wait=True,
            points_selector=models.Filter(must=[
                FieldCondition(key=key, match=models.MatchValue(value=value))
            ])
        )

    def delete_data_cropwizard(self, key: str, value: str):
        """
        Delete data from CropWizard Qdrant collection.
        """
        if not self.cropwizard_qdrant_client:
            print("⚠️ CropWizard Qdrant not initialized.")
            return None

        return self.cropwizard_qdrant_client.delete(
            collection_name="cropwizard",
            wait=True,
            points_selector=models.Filter(must=[
                FieldCondition(key=key, match=models.MatchValue(value=value))
            ])
        )

    def vector_search_with_filter(self, search_query, course_name, doc_groups: List[str],
                                  user_query_embedding, top_n, disabled_doc_groups: List[str],
                                  public_doc_groups: List[dict], custom_filter: models.Filter):
        """
        Search vector database with a custom (conversation) filter.
        """
        return self.qdrant_client.search(
            collection_name=os.environ["QDRANT_COLLECTION_NAME"],
            query_filter=custom_filter,
            with_vectors=False,
            query_vector=user_query_embedding,
            limit=top_n,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(rescore=False)
            )
        )
