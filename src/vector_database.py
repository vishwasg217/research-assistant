from typing import Literal
import weaviate
from weaviate.classes.query import MetadataQuery
from weaviate.collections.classes.filters import _Filters
import os
from dotenv import load_dotenv
from pydantic_classes import Document

load_dotenv(".env")

class VectorDB:
    def __init__(self, collection_name: str):
        openai_key = os.getenv("OPENAI_API_KEY")
        headers = {
            "X-OpenAI-Api-Key": openai_key,
        }
        self.client = weaviate.connect_to_local(
            headers=headers
        )
        self.collection = self.client.collections.get(collection_name)

    def query(
        self, 
        question: str, 
        filters: _Filters = None,
        top_k: int = 10, 
        search_type: Literal["similarity", "keyword", "hybrid"] = "similarity", 
        alpha: int = None
    ) -> list[Document]:

        if search_type == "similarity":
            response = self.collection.query.near_text(
                query=question,
                filters=filters,
                return_metadata=MetadataQuery(distance=True, certainty=True),
                limit=top_k,
                include_vector=True
            )

        elif search_type == "keyword":
            response = self.collection.query.bm25(
                query=question,
                filters=filters,
                return_metadata=MetadataQuery(score=True, explain_score=True),
                limit=top_k
            )
        
        elif search_type == "hybrid":
            response = self.collection.query.hybrid(
                query=question,
                filters=filters,
                # return_metadata=MetadataQuery(score=True, explain_score=True),
                alpha=alpha,
                limit=top_k
            )

        self.client.close()

        return [
            Document(
                uuid=object.uuid.int,
                paper_url=f"https://arxiv.org/abs/{object.properties['paper_id']}",
                html_url=f"https://ar5iv.labs.arxiv.org/html/{object.properties['paper_id']}",
                **object.properties,
                metadata={k: v for k, v in object.metadata.__dict__.items() if v != None}
            ) for object in response.objects
        ]