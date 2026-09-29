"""Runs benchmark queries against the running API. Start it first: uvicorn app:app"""
import os
import sys

import requests

API_URL = os.getenv("API_URL", "http://localhost:8000/chat")
REFUSAL = "cannot answer based on the provided document"

QUERIES = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "Who won the 2022 FIFA World Cup?",  # must be refused
]


def main() -> int:
    failed = False
    for q in QUERIES:
        r = requests.post(API_URL, json={"query": q}, timeout=60)
        r.raise_for_status()
        data = r.json()
        print(f"\nQ: {q}")
        print(f"A: {data['answer']}")
        print(f"Confidence: {data['confidence_score']} | chunks: {len(data['retrieved_chunks'])}")
        if data["retrieved_chunks"]:
            print(f"Top chunk: {data['retrieved_chunks'][0][:150]!r}...")
        if "FIFA" in q and REFUSAL not in data["answer"].lower():
            print("FAIL: out-of-scope question was not refused")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
