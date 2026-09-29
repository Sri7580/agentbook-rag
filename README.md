# Agentic AI RAG Chatbot

RAG chatbot answering strictly from the Agentic AI eBook.
Stack: Python, LangGraph, Pinecone, FastAPI, Streamlit.
Models: free default is local HuggingFace embeddings (`all-MiniLM-L6-v2`, 384 dims) + Groq (`llama-3.1-8b-instant`). Set `PROVIDER=openai` to use `text-embedding-3-small` (1536 dims) + `gpt-4o-mini`.

## Architecture
```
PDF -> PyPDFLoader -> RecursiveCharacterTextSplitter (1000/200) -> embeddings -> Pinecone (cosine)

question -> [retrieve: top-k chunks + cosine score] -> [generate: context-only prompt] -> answer + chunks + confidence
```
- **Grounding:** the prompt allows only the retrieved context. If the best chunk's cosine similarity is below `MIN_RELEVANCE` (default 0.25), the LLM call is skipped and the fixed refusal is returned.
- **Confidence score:** cosine similarity of the top retrieved chunk (0-1). It is `0.0` whenever the system refuses.

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # add GROQ_API_KEY and PINECONE_API_KEY (or OPENAI_API_KEY with PROVIDER=openai)
```

## 1. Ingest
```bash
python -m src.ingestion --download   # or place the PDF at data/Ebook-Agentic-AI.pdf
```
Creates the Pinecone index if missing, then chunks and upserts the eBook.

## 2. Run the API
```bash
uvicorn app:app --reload
```
```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"query": "What is Agentic AI according to the eBook?"}'
```
Response:
```json
{"answer": "...", "retrieved_chunks": ["..."], "confidence_score": 0.61}
```

## 3. Optional Streamlit UI
```bash
streamlit run streamlit_app.py
```

## 4. Test
```bash
python tests_sample_queries.py
```
Runs 5 queries, including an out-of-scope one (2022 FIFA World Cup) that must be refused.

## Configuration
| Variable | Default | Purpose |
|---|---|---|
| `PINECONE_INDEX_NAME` | `agentic-ai-index` | Index name |
| `TOP_K` | `4` | Chunks retrieved |
| `MIN_RELEVANCE` | `0.25` | Refusal threshold; tune after checking scores on your queries |
| `PROVIDER` | `free` | `free` (HuggingFace + Groq) or `openai` |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Generation model |

## Structure
```
rag-agentic-ai/
├── data/Ebook-Agentic-AI.pdf
├── src/{__init__,ingestion,graph,config}.py
├── app.py
├── streamlit_app.py
├── tests_sample_queries.py
├── requirements.txt
├── .env.example
└── README.md
```
