import streamlit as st
from src import VectorDB, Reranker
from src.engines import ResearchEngine
from src.main import summarize_paper


def format_datetime(dt):
    formatted_date = dt.strftime(f'%d %B, %Y')
    return formatted_date


st.title("Research Assistant")

if 'query' not in st.session_state:
    st.session_state.query = None

if 'response' not in st.session_state:
    st.session_state.response = None

if 'summaries' not in st.session_state:
    st.session_state.summaries = {}

vector_db = VectorDB(collection_name="Paper")
reranker = Reranker(model="rerank-english-v3.0", rank_fields=['title', 'abstract'])
engine = ResearchEngine(vector_db=vector_db, reranker=reranker)

st.session_state.query = st.text_input(label="Enter your research query")

if st.button("Search"):
    if st.session_state.query:
        st.session_state.response = engine.query(
            question=st.session_state.query, 
            search_type="hybrid",
            alpha=0.5,
            top_n=5,
            max_words=30
        )


if st.session_state.response:
    st.markdown(st.session_state.response.content)
    st.subheader("Related Papers")

    columns = st.multiselect(
        label="Select the sections you want to summarize",
        options=['introduction', 'abstract', 'methodology', 'findings', 'results'],
        default=['introduction', 'findings', 'results']
    ) 

    for paper in st.session_state.response.context:
        with st.container(border=True):
            title = paper.title.replace('\n', '')
            st.markdown(f"### [{title}]({paper.paper_url})\n")
            st.markdown(f"{paper.html_url}")
            col1, col2, col3 = st.columns(3)
            col1.markdown(f"\n##### Published Date:\n{format_datetime(paper.create_date)}")
            col2.markdown(f"**Authors**\n\n{', '.join(paper.authors)}")
            col3.markdown(f"**Categories**\n\n{', '.join(paper.categories)}")
            st.markdown(f"\n##### Abstract:\n{paper.abstract}")

            if st.button("Summarize", key=paper.paper_id):
                for column in columns:
                    st.session_state.summaries[paper.paper_id][column] = summarize_paper(paper, column).summary

            if paper.paper_id in st.session_state.summaries:     
                for column, summary in st.session_state.summaries[paper.paper_id].items():
                    st.markdown(f"\n##### {column.capitalize()}:\n{summary}")

        

