

## 📄🤖 Local RAG Architectures for AI Tutoring at WU Vienna

This project explores and compares **three RAG (Retrieval-Augmented Generation) architectures** for querying course-specific academic documents — all running **locally and privately** using Streamlit, ChromaDB, and Ollama.

---

🔹 **Naive RAG** serves as the baseline. Documents are chunked **by page**, the top-k most similar chunks are retrieved via vector search, and the results are passed directly to the LLM as flat context. Simple, fast, and easy to understand — but limited in precision.

🔬 **DeepDoc RAG** improves on this by using **layout-aware document parsing** (via `unstructured`) to detect and separate structural elements like titles, paragraphs, and tables before indexing. This enables smarter, more targeted retrieval through **metadata filtering** — so the model isn't distracted by irrelevant document noise.

📊 **TableBook RAG** treats **tables as first-class citizens**. Using `pdfplumber`, prose and tabular data are extracted and stored separately. At query time, **two parallel retrieval queries** are run — one for text, one for tables — and the results are merged to give the LLM richer, more structured context.

---

⚙️ All three architectures share the **same Streamlit UI, ChromaDB vector store, and Ollama LLM backend**. The key differences lie entirely in how documents are **chunked during ingestion** and how context is **assembled during retrieval**. All file formats supported: `pdf`, `docx`, `pptx`, `txt`, `xlsx`, `xls`.

---

## 🛠️ Environment Setup

Each RAG architecture runs in its own isolated Conda environment to avoid dependency conflicts. All environments use **Python 3.11** and **CPU-only PyTorch**.

> ⚠️ **Important:** Always install `sentence-transformers==2.7.0` and `transformers==4.41.0` together. Newer versions cause a `LRScheduler` / `nn` compatibility error with the current version of ChromaDB.

> ⚠️ **Important:** Always pin `numpy==1.26.4`. Newer versions of numpy cause a `Numpy is not available` error with ChromaDB and sentence-transformers.

### 🔹 Naive RAG — `envnaive`
```bash
conda create -n envnaive python=3.11 -y
conda activate envnaive
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install "sentence-transformers==2.7.0" "transformers==4.41.0"
pip install "numpy==1.26.4"
pip install streamlit chromadb pymupdf pytesseract pandas python-docx python-pptx requests openpyxl
```

### 🔬 DeepDoc RAG — `envdeepdoc`
```bash
conda create -n envdeepdoc python=3.11 -y
conda activate envdeepdoc
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install "sentence-transformers==2.7.0" "transformers==4.41.0"
pip install "numpy==1.26.4"
conda install -c conda-forge pikepdf llvmlite numba -y
pip install streamlit chromadb pymupdf pytesseract pandas python-docx python-pptx requests openpyxl "unstructured[pdf]" "unstructured[docx]" "unstructured[pptx]"
```

### 📊 TableBook RAG — `envtablebook`
```bash
conda create -n envtablebook python=3.11 -y
conda activate envtablebook
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install "sentence-transformers==2.7.0" "transformers==4.41.0"
pip install "numpy==1.26.4"
pip install streamlit chromadb pymupdf pytesseract pandas python-docx python-pptx requests openpyxl pdfplumber
```

---

## 🚀 Running the Apps

Always activate the matching environment before running:

```bash
conda activate envnaive && streamlit run naive_app.py
conda activate envdeepdoc && streamlit run deepdoc_app.py
conda activate envtablebook && streamlit run tablebook_app.py
```
