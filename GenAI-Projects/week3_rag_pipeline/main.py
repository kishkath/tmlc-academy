"""
Flask REST API for RAG Legal Assistant.

Endpoints:
GET  /health   – Basic service health check.
POST /query    – Submit a legal question and receive an answer with sources & evaluation.
"""

from flask import Flask, request, jsonify

from configurations.config import CHROMA_PERSIST_DIR
from loggers.logger import logger
from data.dataset_loader import load_dataset_texts
from rag.retriever import HybridRetriever, LCCompatibleHybridRetriever
from rag.chain_builder import build_rag_chain
from rag.evaluator import evaluate_answer

# Initialize Flask app
app = Flask(__name__)

logger.info("Initializing RAG system for Flask app...")
import os

# --- One-time Initialization ---
if os.path.exists(CHROMA_PERSIST_DIR) and any(
    f.endswith(".sqlite3") or f.endswith(".sqlite") for f in os.listdir(CHROMA_PERSIST_DIR)
):
    logger.info(f"Found existing Chroma DB in {CHROMA_PERSIST_DIR}, using it directly.")
    TEXTS = None  # No need to rebuild corpus
else:
    logger.info("No existing Chroma DB found, loading dataset and creating embeddings.")
    TEXTS, METADATAS = load_dataset_texts()

# RETRIEVER = LCCompatibleHybridRetriever(hybrid=HybridRetriever(texts_corpus=TEXTS))
hybrid = HybridRetriever(texts_corpus=TEXTS)
RETRIEVER = LCCompatibleHybridRetriever(hybrid=hybrid)  # ✅ Wrap for LangChain
QA_CHAIN = build_rag_chain()

logger.info("RAG system initialized successfully.")


@app.route("/health", methods=["GET"])
def health_check():
    """Simple health-check endpoint."""
    return jsonify({"status": "ok", "message": "RAG Legal Assistant API running"}), 200


@app.route("/query", methods=["POST"])
def query():
    """
    Accepts JSON: {"question": "..."}
    Returns JSON with answer, sources, and evaluation.
    """
    try:
        data = request.get_json(force=True)
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "Missing 'question' field"}), 400

        logger.info("Received question: %s", question)

        # Run retrieval and generation
        # result = QA_CHAIN(question)
        try:
            result = QA_CHAIN.invoke({"query": question})
        except TypeError:
            # fallback for older LangChain RetrievalQA objects
            result = QA_CHAIN({"query": question})

        answer = result["result"]
        sources = result.get("source_documents", [])

        # Evaluate answer quality
        evaluation = evaluate_answer(answer, sources, question)

        response = {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "page_content": getattr(s, "page_content", ""),
                    "metadata": getattr(s, "metadata", {}),
                }
                for s in sources
            ],
            "evaluation": evaluation,
        }

        return jsonify(response), 200

    except Exception as e:
        logger.exception("Error processing query request.")
        return jsonify({"error": str(e)}), 500


def create_app():
    """Factory function for WSGI servers (e.g., Gunicorn)."""
    return app


if __name__ == "__main__":
    # Local dev mode
    app.run(host="0.0.0.0", port=8000, debug=False)
