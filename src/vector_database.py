from typing import Optional
from typing import Literal
import weaviate
from weaviate.classes.query import MetadataQuery
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from datetime import datetime

load_dotenv(".env")

class Document(BaseModel):
    uuid: int
    title: str
    abstract: str
    categories: list[str]
    authors: list[str]
    create_date: datetime
    update_date: datetime
    paper_url: str
    html_url: str
    paper_id: str
    doi: Optional[str] = None
    report_no: Optional[str] = None
    journal_ref: Optional[str] = None
    license: Optional[str] = None
    comments: Optional[str] = None
    metadata: Optional[dict] = None


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

    def query(self, question: str, top_k: int = 10, query_type: Literal["similarity", "keyword", "hybrid"] = "similarity", alpha: int = None):

        if query_type == "similarity":
            response = self.collection.query.near_text(
                query=question,
                return_metadata=MetadataQuery(distance=True, certainty=True),
                limit=top_k,
                include_vector=True
            )

        elif query_type == "keyword":
            response = self.collection.query.bm25(
                query=question,
                return_metadata=MetadataQuery(score=True, explain_score=True),
                limit=top_k
            )
        
        elif query_type == "hybrid":
            response = self.collection.query.hybrid(
                query=question,
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

if __name__ == "__main__":
    db = VectorDB("Paper")
    response = db.query(question="transformer model")
    print(response)