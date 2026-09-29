"""Optional UI. Run: streamlit run streamlit_app.py"""
import streamlit as st

from src import config
from src.graph import build_rag_graph

st.set_page_config(page_title="Agentic AI RAG Chatbot", layout="wide")


@st.cache_resource
def get_graph():
    return build_rag_graph(config.PINECONE_INDEX_NAME)


st.title("Agentic AI eBook Chatbot")
query = st.chat_input("Ask a question about the eBook")

if query:
    with st.spinner("Thinking..."):
        result = get_graph().invoke(
            {"question": query, "context": [], "answer": "", "score": 0.0}
        )
    chat_col, side_col = st.columns([2, 1])
    with chat_col:
        st.chat_message("user").write(query)
        st.chat_message("assistant").write(result["answer"])
    with side_col:
        st.metric("Confidence", f"{result['score']:.2f}")
        st.subheader("Retrieved chunks")
        for i, chunk in enumerate(result["context"], 1):
            with st.expander(f"Chunk {i}"):
                st.write(chunk)
