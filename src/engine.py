from textwrap import dedent
from openai import OpenAI
from dotenv import load_dotenv
from vector_database import VectorDB
from reranker import Reranker
from typing import Literal
import json


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
    {{"response": "provide the response using proper markdown formatting."}}

    ## Response
"""


class Engine:
    def __init__(self, vector_db: VectorDB, reranker: Reranker = None):
        self.client = OpenAI() 
        self.vector_db = vector_db
        self.reranker = reranker

    def retrieve(
        self, 
        question: str, 
        top_k: int = 10, 
        search_type: Literal["similarity", "keyword", "hybrid"] = "similarity"
    ):
        documents = self.vector_db.query(question=question, top_k=top_k, search_type=search_type)
        
        metric_avg = documents[0].metadata
        for k, v in metric_avg.items():
            for doc in documents[1:]:
                metric_avg[k] += doc.metadata[k]
            metric_avg[k] /= len(documents)

        return documents, metric_avg

    def query(
        self, 
        question: str, 
        level: Literal["beginner", "intermediate", "expert"] = "beginner",
        max_words: int = 100, 
        rerank_n: int = None,
    ) -> str:
        documents, metric_avg = self.retrieve(question=question)
        print(f"Metrics: {metric_avg}")
        if rerank_n and self.reranker is not None:
            documents = self.reranker.rerank(question=question, documents=documents, top_n=rerank_n)

        prompt = dedent(PROMPT_TEMPLATE).format(
            level=level,
            question=question,
            max_words=max_words,
            context=[{"title": doc.title, "abstract": doc.abstract} for doc in documents]
        )

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "you are an assistant who responds in json format"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)["response"]
    







        
        