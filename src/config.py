import os

from dotenv import load_dotenv

load_dotenv()

PDF_PATH = "data/Ebook-Agentic-AI.pdf"
PDF_DRIVE_ID = "15VLphKcY23_fpYxN62UEQRri_psRVfP9"

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

PROVIDER = os.getenv("PROVIDER", "free").lower()
if PROVIDER == "openai":
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIM = 1536
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
else:
    EMBEDDING_MODEL = "multilingual-e5-large"
    EMBEDDING_DIM = 1024
    LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = int(os.getenv("TOP_K", "4"))


MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.25"))

REFUSAL = "I cannot answer based on the provided document."
