# ⚡ AuditMind AI — 10-K Compliance & Financial Audit Engine

🔗 **Live Demo:** [auditmind-ai.streamlit.app](https://auditmind-ai.streamlit.app)

AuditMind is an enterprise-grade Retrieval-Augmented Generation (RAG) system built to audit financial SEC 10-K filings, verify revenue streams, trace performance obligations, and provide precise source page citations with zero hallucination.

---

## 🚀 Key Features

* **High-Precision Document Processing:** Chunked SEC 10-K financial reports using `RecursiveCharacterTextSplitter`.
* **Vector Embeddings & Semantic Search:** Local vector embeddings indexed and retrieved via ChromaDB vector store.
* **Auditing Guardrails & Source Citations:** Zero-hallucination prompt constraints with exact page number citations.
* **Dual-Interface Layer:** Production REST endpoints using FastAPI alongside an interactive Streamlit UI dashboard.
* **Automated Evaluation Suite:** Custom benchmark verification testing domain guardrails and ground truth accuracy.

---

## 🛠️ Tech Stack & Architecture

* **LLM Engine:** Google Gemini (`gemini-3.6-flash`)
* **Framework:** LangChain, FastAPI, Streamlit
* **Vector Store:** ChromaDB
* **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
* **Environment:** Python 3.14

---

## 📦 Setup & Execution

1. **Activate Virtual Environment & Install Dependencies:**
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt

2. **Configure API Keys:**
    ```bash
    GEMINI_API_KEY=your_gemini_api_key

3. **Ingest Financial Documents:**
    ```bash
    python -m app.ingestion

4. **Launch Interactive Streamlit App:**
    ```bash
    streamlit run app/ui.py

5. **Run Accuracy Benchmarks:**
    ```bash
    python evals/test_rag.py