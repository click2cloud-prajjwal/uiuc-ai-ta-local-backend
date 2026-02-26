import os
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.http import models

# Load .env from current directory
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION = os.getenv("QDRANT_COLLECTION_NAME")

if not QDRANT_URL or not COLLECTION:
    raise RuntimeError("Missing QDRANT_URL or QDRANT_COLLECTION_NAME in .env")

client = QdrantClient(
    url=QDRANT_URL,
    port=QDRANT_PORT,
)

courses_to_delete = ["adt", "test", "localhost"]

for course in courses_to_delete:
    print(f"Deleting course: {course}")
    client.delete(
        collection_name=COLLECTION,
        points_selector=models.Filter(
            must=[
                models.FieldCondition(
                    key="course_name",
                    match=models.MatchValue(value=course)
                )
            ]
        ),
        wait=True
    )

print("✅ Deletion completed")
