from .pydantic_classes import Paper, Document
from .vector_database import VectorDB
from .reranker import Reranker
from .engines import ResearchEngine, SummaryEngine
from .web_loader import WebLoader


def summarize_paper(paper: Document, column: str):
    web_loader = WebLoader()
    paper_chunks = web_loader.load_paper(paper.html_url)
    summary_engine = SummaryEngine()
    
    summary = summary_engine.summarize(
        paper=Paper(
            title=paper.title,
            abstract=paper.abstract,
            chunks_content=paper_chunks
        ),
        column=column,
        max_words=50
    )

    return summary


if __name__ == "__main__":
    vector_db = VectorDB(collection_name="Paper")
    reranker = Reranker(model="rerank-english-v3.0", rank_fields=['title', 'abstract'])
    engine = ResearchEngine(vector_db=vector_db, reranker=reranker)

    question = "How do GPT models compare to BERT models for classification tasks?"
    response = engine.query(question=question, level="intermediate", top_n=5, max_words=50)
    print(response.content)

    paper = response.context[2]
    columns = ['introduction', 'abstract', 'methodology', 'findings', 'results']
    summaries = summarize_paper(paper, columns)
    print(f"Summaries:\n")
    print(summaries)
