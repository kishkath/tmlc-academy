"""
Retriever implementations: dense (Chroma), sparse (BM25), and hybrid.
Ensures output is a list of plain text strings for embeddings.
"""

from typing import List, Optional
from langchain.docstore.document import Document
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from configurations.config import (
    EMBEDDING_MODEL_NAME,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    TOP_K
)
from loggers.logger import logger

# Optional BM25
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


class DenseRetriever:
    """Dense retrieval using Chroma embeddings."""

    def __init__(self, top_k: int = TOP_K):
        self.top_k = top_k
        self.embed_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=self.embed_model
        )

    def get_relevant_documents(self, query: str) -> List[str]:
        """Return top-k relevant documents as plain strings."""
        try:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})
            docs = retriever.get_relevant_documents(query)
            # Return only text content
            return [d.page_content for d in docs if hasattr(d, "page_content")]
        except Exception:
            logger.exception("Dense retrieval failed")
            return []


class HybridRetriever:
    """
    Combines dense (Chroma) and optional sparse (BM25) retrieval.
    Always returns list of plain strings.
    """

    def __init__(self, texts_corpus: Optional[List[str]] = None, top_k: int = TOP_K):
        self.top_k = top_k
        self.texts_corpus = texts_corpus or []
        self.dense = DenseRetriever(top_k=top_k * 2)
        self.bm25 = BM25Okapi([t.split() for t in self.texts_corpus]) if BM25Okapi and self.texts_corpus else None

    def retrieve(self, query: str, re_rank: bool = False, llm=None) -> List[str]:
        """Return top-k documents as plain strings."""
        dense_docs = self.dense.get_relevant_documents(query)

        sparse_docs = []
        if self.bm25:
            scores = self.bm25.get_scores(query.split())
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.top_k * 2]
            sparse_docs = [self.texts_corpus[i] for i in top_idx]

        # Merge and keep unique
        combined = list(dict.fromkeys(dense_docs + sparse_docs))
        return combined[:self.top_k]

from langchain.schema import BaseRetriever, Document
from pydantic import BaseModel

# retriever.py
class LCCompatibleHybridRetriever(BaseRetriever):
    """Wrap HybridRetriever for LangChain compatibility."""

    hybrid: HybridRetriever  # Field(...) is optional, just declaring type

    def get_relevant_documents(self, query: str) -> list[Document]:
        texts = self.hybrid.retrieve(query)
        return [Document(page_content=t) for t in texts]



