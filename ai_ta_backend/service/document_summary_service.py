import time
from collections import defaultdict
from typing import Dict
import os
from qdrant_client import QdrantClient


class DocumentSummaryService:
    _cache_data = None
    _cache_time = 0
    _cache_ttl = 60  # seconds

    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            api_key=os.getenv("QDRANT_API_KEY") or None,
            timeout=20,
        )
        self.collection = os.getenv("QDRANT_COLLECTION_NAME")

    def get_summary(self) -> Dict:
        now = time.time()

        # Serve from cache
        if self._cache_data and (now - self._cache_time) < self._cache_ttl:
            return self._cache_data

        course_groups = defaultdict(lambda: defaultdict(set))
        course_total_docs = defaultdict(set)
        overall_docs = set()

        next_offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection,
                with_vectors=False,
                with_payload=True,
                limit=500,
                offset=next_offset,
            )

            for p in points:
                payload = p.payload or {}
                course = payload.get("course_name")
                blob_path = payload.get("blob_path")
                doc_groups = payload.get("doc_groups") or []

                if not course or not blob_path:
                    continue

                overall_docs.add(blob_path)
                course_total_docs[course].add(blob_path)

                for group in doc_groups:
                    course_groups[course][group].add(blob_path)

            if next_offset is None:
                break

        course_names = list(course_groups.keys())

        result = {
            "overall_total_documents": len(overall_docs),
            "total_courses": len(course_names),
            "course_names": course_names,
            "courses": [
                {
                    "course_name": c,
                    "total_documents": len(course_total_docs[c]),
                    "doc_groups": {
                        g: len(docs) for g, docs in course_groups[c].items()
                    }
                }
                for c in course_groups
            ]
        }

        # Update cache
        self._cache_data = result
        self._cache_time = now

        return result
