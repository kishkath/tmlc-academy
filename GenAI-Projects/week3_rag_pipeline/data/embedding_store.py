"""
Embedding generation and Chroma storage utilities.
Handles batching, id generation, and metadata storage.
"""

from typing import List
import uuid
import chromadb
from sentence_transformers import SentenceTransformer
from configurations.config import (
    EMBEDDING_MODEL_NAME,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    CHROMA_BATCH_SIZE
)
from loggers.logger import logger

def create_or_get_chroma_collection(client: chromadb.PersistentClient):
    """
    Return a Chroma collection object; create if not present.
    """
    try:
        try:
            collection = client.get_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            collection = client.create_collection(CHROMA_COLLECTION_NAME)
        return collection
    except Exception as e:
        logger.exception("Failed to get/create Chroma collection")
        raise

def store_embeddings_in_chroma(texts: List[str], metadatas: List[dict]):
    """
    Store text embeddings + metadata in Chroma in batches.
    """
    if len(texts) != len(metadatas):
        raise ValueError("texts and metadatas must be same length")

    # Load encoder
    try:
        encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception:
        logger.exception("Failed loading SentenceTransformer")
        raise

    # Connect to Chroma
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    except Exception:
        logger.exception("Failed connecting to Chroma persistent client")
        raise

    collection = create_or_get_chroma_collection(client)
    total = len(texts)
    logger.info("Storing %d documents into Chroma (batch_size=%d)", total, CHROMA_BATCH_SIZE)

    for start in range(0, total, CHROMA_BATCH_SIZE):
        end = min(start + CHROMA_BATCH_SIZE, total)
        batch_texts = texts[start:end]
        batch_meta = metadatas[start:end]
        try:
            embeddings = encoder.encode(batch_texts, show_progress_bar=False)
            ids = [str(uuid.uuid4()) for _ in batch_texts]
            collection.add(ids=ids, documents=batch_texts, embeddings=embeddings, metadatas=batch_meta)
            logger.info("Stored batch %d (%d documents)", start // CHROMA_BATCH_SIZE + 1, len(batch_texts))
        except Exception:
            logger.exception("Failed to store batch %d-%d", start, end)
            raise

    logger.info("Completed storing embeddings.")
