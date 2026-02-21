from langchain.chains import RetrievalQA, LLMChain, StuffDocumentsChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

from configurations.config import LLM_MODEL_NAME, LLM_TEMPERATURE, OPENAI_API_KEY, TOP_K, EMBEDDING_MODEL_NAME, \
    CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR
from data.dataset_loader import load_dataset_texts
from loggers.logger import logger
import os

import os
import logging
from typing import List, Tuple, Dict, Any

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.schema import BaseRetriever
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# # Import your config variables (assumes config.py in same package)
# from config import (
#     OPENAI_API_KEY,
#     LLM_MODEL_NAME,
#     LLM_TEMPERATURE,
#     TOP_K,
#     CHROMA_COLLECTION_NAME,
#     CHROMA_PERSIST_DIR,
#     EMBEDDING_MODEL_NAME,
# )

# Basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#
# def build_rag_chain(retriever, top_k: int = TOP_K):
#     """
#     Build and return a configured RetrievalQA chain using the provided retriever.
#     Compatible with LangChain >=0.3.0.
#     """
#     if OPENAI_API_KEY:
#         os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
#
#     # Instantiate Chat LLM
#     llm = ChatOpenAI(
#         model_name=LLM_MODEL_NAME,
#         temperature=LLM_TEMPERATURE,
#         openai_api_key=OPENAI_API_KEY
#     )
#
#     # --- Define prompt templates ---
#     system_prompt = (
#         "You are a precise legal assistant whose job is to answer user legal questions using only the provided context. "
#         "Use inline citations [1], [2] referencing the provided context. "
#         "If the answer is not in the context, say 'I don't know based on the available data.'"
#     )
#
#     human_template = (
#         "Context:\n{context}\n\n"
#         "Question:\n{question}\n\n"
#         "Respond with citations and a 'Sources:' line."
#     )
#
#     prompt = ChatPromptTemplate.from_messages([
#         SystemMessagePromptTemplate.from_template(system_prompt),
#         HumanMessagePromptTemplate.from_template(human_template),
#     ])
#
#     # # --- Build LLMChain manually ---
#     # llm_chain = LLMChain(
#     #     llm=llm,
#     #     prompt=prompt
#     # )
#     #
#     # # --- Combine chain ---
#     # combine_documents_chain = StuffDocumentsChain(
#     #     llm_chain=llm_chain,
#     #     document_variable_name="context"
#     # )
#
#     # --- Final RetrievalQA ---
#     combine_docs_chain = create_stuff_documents_chain(llm, prompt)
#     qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
#
#     # qa_chain = RetrievalQA(
#     #     retriever=retriever,
#     #     combine_documents_chain=combine_documents_chain,
#     #     return_source_documents=True
#     # )
#
#     logger.info("Built RetrievalQA chain (manual LLMChain) for LangChain 0.3.23.")
#     return qa_chain

# ------------------------------
# Build RAG chain with hybrid search & enriched system prompt
# ------------------------------
def build_rag_chain(top_k: int = TOP_K, re_rank: bool = True) -> RetrievalQA:
    """
    Build LangChain RetrievalQA pipeline using hybrid search (dense + sparse) with optional LLM re-ranking.
    System prompt enforces:
      - Use ONLY the provided context to answer.
      - Inline citations [1],[2] for retrieved chunks.
      - If answer cannot be supported, respond: "I don't know based on the available data."
      - Provide concise 2-4 sentence answer followed by Sources: line.
    """
    from rank_bm25 import BM25Okapi
    from langchain.schema import Document

    # ------------------------------
    # Load all texts + metadata for sparse retriever (BM25)
    # ------------------------------
    texts, metadatas = load_dataset_texts()
    tokenized_corpus = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # ------------------------------
    # Dense retriever (Chroma)
    # ------------------------------
    embed_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embed_model,
    )
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k*2})  # get more to merge

    # ------------------------------
    # Ensure API key
    # ------------------------------
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # LLM instance
    llm = ChatOpenAI(
        model_name=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY,
    )

    # ------------------------------
    # System + human prompts
    # ------------------------------
    system_prompt = (
        "You are a precise legal assistant whose job is to answer user legal questions using only the provided context. "
        "BEHAVIOR RULES:\n"
        "1) Use ONLY the information present in the 'Context' to produce answers. Do not invent facts.\n"
        "2) Append inline citation [1],[2], ... for facts from context.\n"
        "3) If information is missing, respond: \"I don't know based on the available data.\"\n"
        "4) Concise 2-4 sentence answer followed by Sources: line with metadata.\n"
        "5) Conflicting sources should be noted.\n"
        "6) Neutral, formal tone appropriate for legal info. Not legal advice."
    )
    human_template = "Context:\n{context}\n\nQuestion:\n{question}\n\nRespond with citations and a Sources: line."
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template(human_template),
    ])

    # ------------------------------
    # Custom hybrid retriever
    # ------------------------------
    def hybrid_retriever(query: str) -> List[Document]:
        # Sparse top results
        query_tokens = query.split()
        sparse_scores = bm25.get_scores(query_tokens)
        top_sparse_indices = sparse_scores.argsort()[-top_k*2:][::-1]
        sparse_docs = [
            Document(page_content=texts[i], metadata=metadatas[i]) for i in top_sparse_indices
        ]

        # Dense top results
        dense_docs = dense_retriever.get_relevant_documents(query)

        # Merge & deduplicate by text
        combined = {doc.page_content: doc for doc in sparse_docs + dense_docs}
        merged_docs = list(combined.values())

        # Optional re-ranking via LLM
        if re_rank and merged_docs:
            rerank_prompt = "Rank the following documents by relevance to the query. Return indices in order of relevance."
            docs_text = "\n---\n".join([f"{i}: {d.page_content[:300]}" for i,d in enumerate(merged_docs)])
            full_prompt = f"Query: {query}\n\nDocuments:\n{docs_text}\n\n{rerank_prompt}"
            ranking_resp = llm([HumanMessage(content=full_prompt)])
            ranking_resp_text = ranking_resp.content
            # Try to parse numbers from LLM response
            import re
            indices = [int(x) for x in re.findall(r"\d+", str(ranking_resp_text))]
            if indices:
                merged_docs = [merged_docs[i] for i in indices if i < len(merged_docs)]

        # Return top_k
        return merged_docs[:top_k]

    # ------------------------------
    # Build RetrievalQA using our hybrid retriever
    # ------------------------------
    from langchain.schema import BaseRetriever
    class HybridRetriever(BaseRetriever):
        def get_relevant_documents(self, query: str) -> List[Document]:
            return hybrid_retriever(query)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=HybridRetriever(),
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
    )

    logging.info("RAG pipeline built with hybrid search + system prompt.")
    return qa_chain
