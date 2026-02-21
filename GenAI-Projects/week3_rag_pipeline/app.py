## Streamlit Deployable

"""
Streamlit app for RAG Legal Assistant with automatic Chroma embedding storage.
"""
import sys
import os

# ------------------------------
# Ensure repo root is in PYTHONPATH
# ------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))  # week_3_rag_pipeline folder
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)


import os
import streamlit as st
from loggers.logger import logger
from configurations.config import CHROMA_PERSIST_DIR, OPENAI_API_KEY, TOP_K
from data.dataset_loader import load_dataset_texts
from data.embedding_store import store_embeddings_in_chroma
from rag.chain_builder import build_rag_chain
from rag.retriever import HybridRetriever, LCCompatibleHybridRetriever
from rag.evaluator import evaluate_answer

st.set_page_config(page_title="RAG Legal Assistant", layout="wide")
st.title("📚 RAG Legal Assistant")

# ------------------------------
# Step 1: Ensure embeddings exist in Chroma
# ------------------------------
@st.cache_resource(show_spinner=False)
def init_chroma_embeddings():
    """Create embeddings in Chroma DB if not already persisted."""
    if not os.path.exists(CHROMA_PERSIST_DIR) or not any(f.endswith(".sqlite3") for f in os.listdir(CHROMA_PERSIST_DIR)):
        logger.info("No existing Chroma DB, creating embeddings...")
        texts, metadatas = load_dataset_texts()
        store_embeddings_in_chroma(texts, metadatas)
        return texts, metadatas
    else:
        logger.info("Chroma DB already exists, skipping embedding storage.")
        return None, None

# ------------------------------
# Step 2: Initialize QA chain
# ------------------------------
@st.cache_resource(show_spinner=False)
def init_qa_chain():
    """Build and return the RetrievalQA chain using hybrid retriever."""
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    texts, metadatas = init_chroma_embeddings()

    # Initialize retriever if we have texts loaded
    if texts:
        hybrid = HybridRetriever(texts_corpus=texts)
        retriever = LCCompatibleHybridRetriever(hybrid=hybrid)
    else:
        retriever = None  # QA chain uses its own internal retriever

    qa_chain = build_rag_chain(top_k=TOP_K)
    logger.info("RAG system initialized successfully.")
    return qa_chain

QA_CHAIN = init_qa_chain()

# ------------------------------
# Streamlit UI
# ------------------------------
question = st.text_area("Enter your legal question:", height=100)

if st.button("Get Answer") and question.strip():
    with st.spinner("Retrieving answer..."):
        try:
            # Run retrieval + generation
            try:
                result = QA_CHAIN.invoke({"query": question})
            except TypeError:
                # fallback for older LangChain versions
                result = QA_CHAIN({"query": question})

            answer = result.get("result", "")
            sources = result.get("source_documents", [])

            # Evaluate answer
            evaluation = evaluate_answer(answer, sources, question)

            # Display
            st.subheader("✅ Answer")
            st.write(answer)

            st.subheader("📄 Sources")
            for i, s in enumerate(sources, 1):
                st.markdown(f"**[{i}]** {getattr(s, 'page_content', '')[:300]}...  | Metadata: {getattr(s, 'metadata', {})}")

            st.subheader("📝 Evaluation")
            st.json(evaluation)

        except Exception as e:
            logger.exception("Error processing query.")
            st.error(f"Error: {str(e)}")
