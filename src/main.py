from .vector_database import VectorDB
from .reranker import Reranker
from .engines.research_engine import ResearchEngine

if __name__ == "__main__":
    vector_db = VectorDB(collection_name="Paper")
    reranker = Reranker(model="rerank-english-v3.0", rank_fields=['title', 'abstract'])
    engine = ResearchEngine(vector_db=vector_db, reranker=reranker)

    question = "How do GPT models compare to BERT models for classification tasks?"
    response = engine.query(question=question, level="intermediate", top_n=5, max_words=200)
    print(response.content)