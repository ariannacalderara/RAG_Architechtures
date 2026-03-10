## Local RAG Architectures for an AI Tutor Application for the Department of Information Systems and Operations Management at WU Vienna

This project explores and compares **three RAG (Retrieval-Augmented Generation) architectures** for querying university documents, all running **locally and privately** using Streamlit, ChromaDB, and Ollama.

---

🔹 **Naive RAG** serves as the baseline. Documents are chunked **by page**, the top-k most similar chunks are retrieved via vector search, and the results are passed directly to the LLM as flat context. Simple, fast, and easy to understand, but limited in precision.

🔬 **DeepDoc RAG** improves on this by using **layout-aware document parsing** (via `unstructured`) to detect and separate structural elements like titles, paragraphs, and tables before indexing. This enables smarter, more targeted retrieval through **metadata filtering**, so the model isn't distracted by irrelevant document noise.

📊 **TableBook RAG** treats **tables as first-class citizens**. Using `pdfplumber`, prose and tabular data are extracted and stored separately. At query time, **two parallel retrieval queries** are run, one for text, one for tables — and the results are merged to give the LLM richer, more structured context.

---

⚙️ All three architectures share the **same Streamlit UI, ChromaDB vector store, and Ollama LLM backend**. The key differences lie entirely in how documents are **chunked during ingestion** and how context is **assembled during retrieval**.
