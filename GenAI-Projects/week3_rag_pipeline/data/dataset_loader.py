"""
Dataset loading and preprocessing utilities.
- Loads Hugging Face dataset
- Builds combined text documents and per-document metadata
"""

from typing import List, Tuple
from datasets import load_dataset
from configurations.config import DATASET_NAME, DATASET_SPLIT
from loggers.logger import logger

def load_dataset_texts(dataset_name: str = DATASET_NAME, split: str = DATASET_SPLIT) -> Tuple[List[str], List[dict]]:
    """
    Load Hugging Face dataset and combine act_title, section, and law into one text per row.

    Returns:
        texts: list of combined textual documents
        metadatas: list of metadata dicts with keys like 'act_title' and 'section'
    """
    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        logger.exception("Failed to load dataset")
        raise

    texts = []
    metadatas = []
    # Expect columns: act_title, section, law
    for row in dataset:
        try:
            act = row.get("act_title", "")
            section = row.get("section", "")
            law_text = row.get("law", "")
            if not law_text:
                continue
            doc = f"Act: {act}\nSection: {section}\nLaw: {law_text}"
            texts.append(doc)
            metadatas.append({"act_title": act, "section": section})
        except Exception:
            logger.exception("Skipping malformed row in dataset")

    logger.info("Loaded %d documents from dataset %s (split=%s)", len(texts), dataset_name, split)
    return texts, metadatas
