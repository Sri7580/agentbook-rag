"""LangGraph workflow: retrieve -> generate, strictly grounded in the eBook."""
from typing import List, TypedDict

from langchain_pinecone import PineconeVectorStore
from langgraph.graph import END, START, StateGraph

from src import config
from src.models import get_embeddings, get_llm


class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float  # cosine similarity of the best retrieved chunk (0-1)


SYSTEM_PROMPT = f"""You are a strict question-answering assistant.
Answer ONLY using the context provided by the user message. Do not use outside knowledge.
If the context does not contain enough information to answer, reply exactly:
"{config.REFUSAL}"
Keep answers clear and concise."""


def build_rag_graph(index_name: str = config.PINECONE_INDEX_NAME):
    embeddings = get_embeddings()
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    llm = get_llm()

    def retrieve_node(state: AgentState) -> dict:
        results = vectorstore.similarity_search_with_score(state["question"], k=config.TOP_K)
        context = [doc.page_content for doc, _ in results]
        top = max((s for _, s in results), default=0.0)
        return {"context": context, "score": round(max(0.0, min(1.0, float(top))), 4)}

    def generate_node(state: AgentState) -> dict:
        # Out-of-scope guard: skip the LLM when nothing relevant was retrieved.
        if not state["context"] or state["score"] < config.MIN_RELEVANCE:
            return {"answer": config.REFUSAL, "score": 0.0}

        context_str = "\n\n---\n\n".join(state["context"])
        response = llm.invoke(
            [
                ("system", SYSTEM_PROMPT),
                ("human", f"Context:\n{context_str}\n\nQuestion: {state['question']}"),
            ]
        )
        answer = response.content.strip()
        # The model itself declined -> confidence is zero.
        score = 0.0 if config.REFUSAL.lower() in answer.lower() else state["score"]
        return {"answer": answer, "score": score}

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()
