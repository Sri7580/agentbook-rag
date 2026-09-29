"""Model factory. Free default: Pinecone-hosted embeddings + Groq LLM. Set PROVIDER=openai to use OpenAI."""
import os

from src import config


def get_embeddings():
    if config.PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    from langchain_pinecone import PineconeEmbeddings
    return PineconeEmbeddings(
        model=config.EMBEDDING_MODEL,
        pinecone_api_key=os.environ["PINECONE_API_KEY"],
    )


def get_llm():
    if config.PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=config.LLM_MODEL, temperature=0)
    from langchain_groq import ChatGroq
    return ChatGroq(model=config.LLM_MODEL, temperature=0)