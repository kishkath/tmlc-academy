"""
Top-level runner for local development.
Usage:
    python run.py
This script demonstrates how to run ingestion once and then start the app.
"""

from loggers.logger import logger
from data.dataset_loader import load_dataset_texts
from data.embedding_store import store_embeddings_in_chroma
from rag.retriever import HybridRetriever, DenseRetriever
from rag.chain_builder import build_rag_chain

def run_ingest_and_server():
    # 1. Load dataset and build embeddings (only run once or when dataset changes)
    texts, metadatas = load_dataset_texts()
    logger.info("Starting ingestion to Chroma...")
    store_embeddings_in_chroma(texts, metadatas)
    logger.info("Ingestion complete.")

    # 2. Set up retriever + chain for interactive or API use
    hybrid = HybridRetriever(texts_corpus=texts)
    qa_chain = build_rag_chain(retriever=hybrid)

    # 3. For now start interactive loop (we use rag.interactive in Step 2/3)
    # Placeholder: print ready message
    logger.info("RAG pipeline ready. Integrate with Flask (Step 2) or Streamlit (Step 3).")

if __name__ == "__main__":
    run_ingest_and_server()
