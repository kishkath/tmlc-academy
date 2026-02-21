# 🧠 Retrieval-Augmented Generation (RAG) Legal Information Assistant

Deployed Details: https://github.com/kishkath/rag-legal-assistant/blob/main/README.md

## 📘 Overview

This project is an **intelligent Retrieval-Augmented Generation (RAG) pipeline** designed to retrieve, reason, and respond accurately to **legal domain queries** using structured contextual data.

It integrates **semantic search**, **hybrid retrieval**, and **LLM-based reasoning** to answer questions based solely on factual and verifiable information.
The system behaves like a **context-grounded legal assistant** that never hallucinates — it answers *only* using available law data, and transparently cites its sources.

---

## 🎯 Business Problem

Legal professionals, compliance teams, and policy analysts often need to extract precise legal information from massive collections of **acts, sections, and legal documents**.
Traditional keyword search tools return either too many irrelevant results or lack contextual reasoning to interpret complex clauses.

This RAG system bridges that gap by combining:

1. **Semantic understanding** (contextual embeddings)
2. **Document-level reasoning** (LLM comprehension)
3. **Faithfulness validation** (LLM evaluator for factual accuracy)

The result is a **trustworthy, interpretable** assistant capable of answering law-related queries, summarizing sections, or cross-referencing acts — all grounded in verifiable data.

---

## 🧩 System Architecture

### 1. **Dataset Processing Layer**

* Legal datasets (e.g., acts, sections, laws) are loaded and transformed into structured text format.
* Each record includes **metadata** such as `act_title`, `section`, and `law`.
* Text and metadata are stored for traceability and explainability.

### 2. **Vectorization & Storage Layer**

* The processed dataset is embedded using **SentenceTransformer** models.
* These embeddings are stored in **ChromaDB**, a persistent, high-performance vector database.
* Each document is assigned a **unique identifier** with associated metadata for precise retrieval.

### 3. **Retrieval Layer**

* When a query is received, the system retrieves top similar documents using:

  * **Dense retrieval** (semantic similarity using embeddings)
  * **Optional sparse retrieval (BM25)** for keyword relevance
* A **hybrid search** approach ensures both precision and recall, capturing semantic and lexical matches.

### 4. **Reasoning Layer (LLM Integration)**

* A **ChatOpenAI model (e.g., GPT-4o-mini)** processes the query and retrieved documents.
* The system uses a **strict system prompt** that:

  * Enforces use of only retrieved context
  * Cites sources inline using `[1], [2]`
  * Prevents hallucinations or assumptions
  * Produces a concise, factual answer
  * Appends a “Sources” line showing which documents were referenced

### 5. **Evaluation & Faithfulness Layer**

* After generating an answer, the system self-evaluates it using another LLM call.
* The evaluator scores the response for:

  * **Faithfulness** — Is the answer grounded in retrieved context?
  * **Completeness** — Are key legal points covered?
  * **Precision** — Any incorrect or speculative content?
* This ensures reliability, allowing users to trust the output.

### 6. **Interactive CLI Interface**

* Users can query the system in real-time via a **command-line interface**.
* The interface displays:

  * Retrieved document snippets (context)
  * Final answer with citations
  * Faithfulness evaluation score and comments
* This transparency builds user confidence and allows easy debugging.

---

## 🔍 Business Value & Use Cases

### ⚖️ Legal Research

Lawyers, paralegals, and compliance officers can query specific sections of laws, acts, or regulations to get direct, cited answers.
Example:

> “What are the penalties for fraudulent representation under the Companies Act?”

### 🧾 Policy & Compliance

Organizations can integrate the system into internal compliance portals to provide automatic context-aware responses based on regulatory data.

### 🏛️ Government / Legal Tech Platforms

Government portals and legal tech companies can use this architecture to build citizen-facing Q&A systems or judgment summarizers.

### 🧠 Knowledge Auditing & Validation

The evaluation layer allows auditing of generated answers — critical for high-stakes domains like law, healthcare, or finance.

---

## 🧱 Key Design Principles

1. **Explainability**
   Every answer is traceable back to the exact sections used for reasoning.

2. **Faithfulness & Reliability**
   The LLM is explicitly instructed to avoid speculation and confirm facts only from retrieved context.

3. **Modularity**
   Each stage (data loading, embedding, retrieval, reasoning, evaluation) can evolve independently or scale horizontally.

4. **Persistence & Reproducibility**
   All embeddings and metadata are stored persistently in ChromaDB, ensuring reproducible results.

5. **Future Extensibility**
   The architecture supports future upgrades:

   * Re-ranking with LLM-based evaluators
   * Context summarization
   * UI dashboards
   * Multi-lingual datasets
   * Integration with LangSmith for monitoring & tracing

---

## 🚀 Future Roadmap

### 1. **Feature Enhancements**

* Add **hybrid search with re-ranking** using BM25 + dense embeddings.
* Integrate **context summarization** for large context windows.
* Improve **faithfulness evaluation** with automatic scoring and reports.

### 2. **Observability & Analytics**

* Connect with **LangSmith** for trace-level visibility into prompt behavior, latency, and performance.
* Add monitoring dashboards to track accuracy trends and retrieval precision.

### 3. **Frontend Experience**

* Build a **Streamlit or Flask-based web UI** for querying, viewing citations, and inspecting retrieved contexts.
* Enable batch query evaluation and comparison.

### 4. **Model Adaptation**

* Experiment with **domain-specific fine-tuning** (legal embeddings or LLMs).
* Introduce **multi-model routing** — small models for recall, large models for reasoning.

---

## 🗾️ System Flow Diagram (Text-Based)

```
┌────────────────────┐
│  User Query Input  │
└─────────┬──────────┘
          │
          ▼
┌────────────────────────┐
│ Dataset Processing     │
│ - Load legal text data │
│ - Extract metadata     │
└─────────┬──────────────┘
          │
          ▼
┌────────────────────────┐
│ Embedding Generation   │
│ - Encode via Sentence  │
│   Transformer          │
└─────────┬──────────────┘
          │
          ▼
┌────────────────────────┐
│ Vector Storage (Chroma)│
│ - Store embeddings     │
│ - Maintain metadata    │
└─────────┬──────────────┘
          │
          ▼
┌────────────────────────┐
│ Hybrid Retrieval Layer │
│ - BM25 + Dense Search  │
│ - Top-k Context Chunks │
└─────────┬──────────────┘
          │
          ▼
┌────────────────────────┐
│ LLM Reasoning Layer    │
│ - Use ChatOpenAI       │
│ - Apply System Prompt  │
│ - Generate Answer w/   │
│   Inline Citations     │
└─────────┬──────────────┘
          │
          ▼
┌────────────────────────┐
│ Faithfulness Evaluation│
│ - LLM Evaluator checks │
│ - Scores answer        │
└─────────┬──────────────┘
          │
          ▼
┌────────────────────────┐
│ Output & Sources       │
│ - Display context,     │
│   answer, evaluation   │
└────────────────────────┘
```

## 🔄 Typical Workflow
User enters a question in the Streamlit UI.
Backend retrieves relevant chunks from the loaded documents (hybrid search).
LLM generates a natural-language answer.
Evaluator LLM verifies the answer’s faithfulness to the retrieved sources.
Frontend displays:
Retrieved passages
Generated answer
Evaluation score and comment

Start the backend API:

python main.py
It runs by default on:
http://127.0.0.1:8000

Launch the Frontend Streamlit app:
streamlit run streamlit_app.py

Then open your browser at:
http://localhost:8501


## 🧰 Technologies Used
Component	Technology / Library
Backend API	Flask
Frontend UI	Streamlit
LLM Integration	LangChain + OpenAI / Azure OpenAI
Retrieval Models	TF-IDF, Embedding-based search
Evaluation	LLM-based JSON scorer
Logging	Python logging module
Environment Mgmt	.env + Config class

---

## 🗳️ Summary

This system demonstrates a **complete, production-grade RAG pipeline** that merges:

* Structured data processing,
* Dense retrieval via embeddings,
* Large language model reasoning,
* Transparent faithfulness validation, and
* User-friendly interaction.

It represents a **foundation for enterprise-grade legal information systems**, capable of scaling to millions of legal documents while maintaining accuracy, explainability, and reliability.

## 👨‍💻 Authors

Developed by Sai Kiran as part of an applied learning exercise in RAG pipelines, modular design, and LLM evaluation.
