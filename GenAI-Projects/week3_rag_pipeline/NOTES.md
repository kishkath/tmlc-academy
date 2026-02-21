"""
RAG Pipeline: Hybrid Retriever with BM25 + Chroma + OpenAI LLM
-------------------------------------------------------------

This module builds a hybrid Retrieval-Augmented Generation (RAG) pipeline that combines
both sparse (BM25 keyword-based) and dense (embedding-based via Chroma) retrieval to
provide accurate, citation-backed responses to user queries.

Core Workflow:
1. **Dataset Loading**: Loads preprocessed text chunks and their metadata.
2. **Sparse Retrieval (BM25)**: Finds documents with matching or similar keywords.
3. **Dense Retrieval (Chroma)**: Uses SentenceTransformer embeddings for semantic similarity search.
4. **Hybrid Combination**: Merges and deduplicates results from both retrievers.
5. **Optional Re-ranking**: Optionally leverages the LLM to re-rank retrieved documents.
6. **LLM Response Generation**: Uses an OpenAI LLM (e.g., GPT-4 or GPT-3.5) to generate
   structured, citation-rich answers guided by a legal-style system prompt.
7. **RetrievalQA Chain**: Returns a LangChain RetrievalQA chain that executes the entire
   retrieval + generation workflow.

Key Components:
- **BM25Okapi** — for sparse, keyword-based relevance scoring.
- **Chroma Vector Store** — for dense, embedding-based semantic retrieval.
- **HybridRetriever** — merges both retrieval strategies for improved recall and precision.
- **LangChain ChatOpenAI** — for final reasoning, synthesis, and answer generation.
- **Prompt Templates** — ensure consistent, context-aware question answering with citations.

Usage:
    qa_chain = build_rag_chain()
    response = qa_chain.run("What are the key clauses in the contract?")
    print(response)

This design ensures both lexical accuracy (via BM25) and semantic understanding (via Chroma),
making it suitable for legal, compliance, and documentation-based applications.
"""
