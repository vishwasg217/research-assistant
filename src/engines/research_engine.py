from textwrap import dedent
from openai import OpenAI
from dotenv import load_dotenv
from ..vector_database import VectorDB
from ..reranker import Reranker
from ..pydantic_classes import Query, EngineResponse
from typing import Literal
import json
from datetime import datetime, timezone
from weaviate.classes.query import Filter

load_dotenv(".env")

PROMPT_TEMPLATE = """ 
    You are a research expert on Artificial Intelligence, Machine Learning and other related concepts.
    Your will be provided with a question and abstracts of related papers on which needs to provide a well constructed and structured answer.

    ## Instructions:
    - Provide an overall concensus and summary based on the abstract provided that answers the question.
    - The response must be no more than {max_words} words
    - Avoid fluff and clichés: Generate a concise answers and avoid words, phrases, and sentences that do not add any substantial value to the response.
    - Tone: needs to be conversational, spartan, use less corporate jargon.
    - Assume that the reader has a {level} level of understanding of the topic, so generate response and use terminology accordingly.
    
    ## Question:
    {question}

    ## Context:
    The context provided below is order from most relevant to least relevant to the question. So use the context to accordingly to structure your response.
    {context}

    ## Response Format:
    {{"response": "MUST provide the response using proper markdown formatting."}}

    ## Response
"""

QUERY_TRANSFORM_PROMPT_TEMPLATE = """ 
    Your task is to transform a given query into a schema/structure given.
    This schema will be used to filter articles based on the given query.

    Here is a description of the parameters:
    - questions: rewrite question such that is more suited for hybrid search(vector search + bm25 search).
    - authors: Only if a particular author(s) of the Paper is mentioned in the query.
    - categories: Mention the most suited particular arxiv paper category(s) to the rewritten query.
    - date_range: Create a date range if there is any mention of time period/date in the query. like a year or month
"""


class ResearchEngine:
    def __init__(self, vector_db: VectorDB, reranker: Reranker = None):
        self.client = OpenAI() 
        self.vector_db = vector_db
        self.reranker = reranker

    def retrieve(
        self, 
        query: Query,
        top_k: int = 10, 
        search_type: Literal["similarity", "keyword", "hybrid"] = "similarity",
        alpha: int = None
    ):
        start_date = None
        end_date = None
        if query.date_range:
            start_date = datetime.strptime(query.date_range.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) if query.date_range.start_date else None
            end_date = datetime.strptime(query.date_range.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) if query.date_range.end_date else None
        
        categories = [cat.value for cat in query.categories] if query.categories else None

        filters_list = []
        if query.authors:
            for auth in query.authors:
                filters_list.append(Filter.by_property("authors").equal(auth))
        if query.categories:
            for cat in categories:
                filters_list.append(Filter.by_property("categories").equal(cat))
        if start_date:
            filters_list.append(Filter.by_property("date").greater_than(start_date))
        if end_date:
            filters_list.append(Filter.by_property("date").less_than(end_date))

        # Combine all filters using the AND (&) operator, if any filters exist
        filters = None
        if filters_list:
            filters = filters_list[0]
            for f in filters_list[1:]:
                filters = filters & f

        documents = self.vector_db.query(
            question=query.question, 
            filters=filters, 
            top_k=top_k, 
            search_type=search_type, 
            alpha=alpha
        )
        
        metric_avg = documents[0].metadata
        for k, v in metric_avg.items():
            for doc in documents[1:]:
                metric_avg[k] += doc.metadata[k]
            metric_avg[k] /= len(documents)

        return documents, metric_avg
    
    def query_transform(self, question: str) -> Query:
        response = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": QUERY_TRANSFORM_PROMPT_TEMPLATE},
                {"role": "user", "content": question}
            ],
            response_format=Query
        )

        return response.choices[0].message.parsed

    def query(
        self, 
        question: str, 
        search_type: Literal["similarity", "keyword", "hybrid"] = "similarity",
        alpha: int = None,
        top_k: int = 10,
        top_n: int = None,
        level: Literal["beginner", "intermediate", "expert"] = "beginner",
        max_words: int = 100, 
    ) -> EngineResponse:
        
        transformed_query = self.query_transform(question)
        print(f"\nTransformed Query: {transformed_query}\n")

        if search_type == "hybrid" and alpha is None:
            raise ValueError("Alpha value must be provided for hybrid search")
        documents, metric_avg = self.retrieve(
            query=transformed_query, 
            top_k=top_k,
            search_type=search_type,
            alpha=alpha
        )
        print(f"Metrics: {metric_avg}\n")
        if top_n and self.reranker is not None:
            reranked_documents = self.reranker.rerank(question=question, documents=documents, top_n=top_k)

        top_docs = top_n if top_n else top_k
        prompt = dedent(PROMPT_TEMPLATE).format(
            level=level,
            question=question,
            max_words=max_words,
            context=[{"title": doc.title, "abstract": doc.abstract} for doc in reranked_documents[:top_docs]],
        )

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "you are an assistant who responds in json format"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        return EngineResponse(
            content=json.loads(response.choices[0].message.content)["response"],
            context=documents,
            metadata={
                "retrieval_metrics": metric_avg
            }
        )