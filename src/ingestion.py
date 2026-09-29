import argparse
import os
import time

import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from src import config
from src.models import get_embeddings


def download_pdf(path: str = config.PDF_PATH) -> str:
    """Fetch the eBook from Google Drive. If it fails, download manually to `path`."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = f"https://drive.google.com/uc?export=download&id={config.PDF_DRIVE_ID}"
    session = requests.Session()
    resp = session.get(url, timeout=60)
    if not resp.content.startswith(b"%PDF"):
        resp = session.get(url + "&confirm=t", timeout=120)
    if not resp.content.startswith(b"%PDF"):
        raise RuntimeError(f"Automatic download failed. Save the PDF manually as {path}.")
    with open(path, "wb") as f:
        f.write(resp.content)
    return path


def ensure_index(index_name: str) -> None:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if index_name in pc.list_indexes().names():
        return
    pc.create_index(
        name=index_name,
        dimension=config.EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(2)


def run_ingestion(pdf_path: str, index_name: str) -> PineconeVectorStore:
    ensure_index(index_name)

    docs = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"Loaded {len(docs)} pages -> {len(chunks)} chunks")

    embeddings = get_embeddings()
    store = PineconeVectorStore.from_documents(
        documents=chunks, embedding=embeddings, index_name=index_name
    )
    print(f"Upserted {len(chunks)} chunks to index '{index_name}'")
    return store


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest the Agentic AI eBook into Pinecone")
    parser.add_argument("--pdf", default=config.PDF_PATH)
    parser.add_argument("--index", default=config.PINECONE_INDEX_NAME)
    parser.add_argument("--download", action="store_true", help="download the PDF first")
    args = parser.parse_args()

    if args.download or not os.path.exists(args.pdf):
        download_pdf(args.pdf)
    run_ingestion(args.pdf, args.index)
