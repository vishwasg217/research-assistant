from .pydantic_classes import Paper, Document
from .vector_database import VectorDB
from .reranker import Reranker
from .engines import ResearchEngine, SummaryEngine
from .web_loader import WebLoader


def summarize_paper(paper: Document):
    web_loader = WebLoader()
    paper_chunks = web_loader.load_paper(paper.html_url)

    print(len(paper_chunks))
    
    summary_engine = SummaryEngine()

    summaries = summary_engine.summarize(
        paper=Paper(
            title=paper.title,
            abstract=paper.abstract,
            chunks_content=paper_chunks
        ),
        columns=['introduction', 'methodology', 'results']
    )

    return summaries


if __name__ == "__main__":
    vector_db = VectorDB(collection_name="Paper")
    reranker = Reranker(model="rerank-english-v3.0", rank_fields=['title', 'abstract'])
    engine = ResearchEngine(vector_db=vector_db, reranker=reranker)

    question = "How do GPT models compare to BERT models for classification tasks?"
    response = engine.query(question=question, level="intermediate", top_n=5, max_words=200)
    print(response.content)

    paper = response.context[0]

    summaries = summarize_paper(paper)
    print(f"Summaries:\n")
    print(summaries)
