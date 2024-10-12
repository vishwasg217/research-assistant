import streamlit as st
from src import VectorDB, Reranker, Engine

st.title("Research Assistant")

if 'response' not in st.session_state:
    st.session_state.response = None

vector_db = VectorDB(collection_name="Paper")
reranker = Reranker(model="rerank-english-v3.0", rank_fields=['title', 'abstract'])
engine = Engine(vector_db=vector_db, reranker=reranker)

query = st.text_input(label="Enter your research query")

if st.button("Search"):
    if query:
        st.session_state.response = engine.query(
            question=query, 
            search_type="hybrid",
            alpha=0.5,
            top_n=5,
            max_words=200
        )
    st.markdown(st.session_state.response.content)
    st.subheader("Related Papers")

    for paper in st.session_state.response.context:
        with st.container(border=True):
            title = paper.title.replace('\n', '')
            st.markdown(f"### [{title}]({paper.paper_url})\n")
            col1, col2 = st.columns(2)
            col1.markdown(f"**Authors**\n\n{', '.join(paper.authors)}")
            col2.markdown(f"**Categories**\n\n{', '.join(paper.categories)}")
            st.markdown(f"\n##### Abstract:\n{paper.abstract}")

