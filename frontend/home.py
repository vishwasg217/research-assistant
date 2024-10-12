import streamlit as st
import pandas as pd
from src import VectorDB, Reranker, Engine

st.title("Research Assistant")

vector_db = VectorDB(collection_name="Paper")
reranker = Reranker(model="rerank-english-v3.0", rank_fields=['title', 'abstract'])
engine = Engine(vector_db=vector_db, reranker=reranker)

query = st.text_input(label="Enter your research query")

if st.button("Search"):
    if query:
        response = engine.query(question=query, max_words=250, top_n=5)
    st.markdown(response.content)
    st.subheader("Related Papers")
    papers_df = pd.DataFrame([doc.model_dump() for doc in response.context])
    print(papers_df.columns)
    papers_df = papers_df[["title", "abstract", "authors", "categories", "paper_url"]]
    st.dataframe(
        data = papers_df,
    )

