import cohere
from dotenv import load_dotenv
import os
from src.pydantic_classes import Document

load_dotenv(".env")

class Reranker:
    def __init__(self, model, rank_fields):
        self.model = model
        self.rank_fields = rank_fields
        self.co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

    def rerank(
        self, 
        question: str, 
        documents: list[Document], 
        top_n: int = 5, 
        return_documents: bool = False
    ) -> list[Document]:
        documents_dict = [doc.model_dump() for doc in documents]
        response = self.co.rerank(
            model=self.model,
            documents=documents_dict,
            query=question,
            rank_fields=self.rank_fields,
            top_n=top_n,
            return_documents=return_documents
        )
        avg_relevance = sum([doc.relevance_score for doc in response.results])/len(response.results)
        print(f"Avg Relevance Score: {avg_relevance}\n")
        reranked_documents = [documents[doc.index] for doc in response.results]
        return reranked_documents