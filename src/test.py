from typing import Literal
import weaviate
from weaviate.classes.query import MetadataQuery
import os
from dotenv import load_dotenv

load_dotenv(".env")

class VectorDB:
    def __init__(self, collection_name: str):
        openai_key = os.getenv("OPENAI_API_KEY")
        headers = {
            "X-OpenAI-Api-Key": openai_key,
        }
        client = weaviate.connect_to_local(
            headers=headers
        )
        self.collection = client.collections.get(collection_name)

    def query(self, question: str, query_type: Literal["similarity", "keyword", "hybrid"], alpha: int = None):

        if query_type == "similarity":
            response = self.collection.query.near_text(
                query=question,
                return_metadata=MetadataQuery(distance=True, certainty=True),
                limit = 10
            )

        elif query_type == "keyword":
            response = self.collection.query.bm25(
                query=question,
                return_metadata=MetadataQuery(score=True, explain_score=True),
                limit = 10
            )

