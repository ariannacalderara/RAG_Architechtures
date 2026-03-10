import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import requests
import fitz
import pytesseract
from PIL import Image
import io
import os
import docx
from pptx import Presentation
import pandas as pd
from unstructured.partition.auto import partition
from unstructured.documents.elements import Table, Title, NarrativeText, ListItem

TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

CHROMA_DIR = "chroma_data"
client = chromadb.PersistentClient(path=CHROMA_DIR)
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name="course_docs", embedding_function=embed_fn)

# ── Extractors ────────────────────────────────────────────────────────────────

def extract_chunks_deepdoc(file_path, file_name):
    elements = partition(filename=file_path)
    
    chunks = []
    current_chunk = []
    current_type = None

    for el in elements:
        text = str(el).strip()
        if not text:
            continue

        el_type = type(el).__name__  # "Title", "NarrativeText", "Table", "ListItem", etc.

        # Start a new chunk when hitting a Title (like a heading)
        if isinstance(el, Title) and current_chunk:
            chunks.append({
                "text": "\n".join([t for t, _ in current_chunk]),
                "type": current_type or "NarrativeText"
            })
            current_chunk = []

        current_chunk.append((text, el_type))
        current_type = el_type

    # Don't forget the last chunk
    if current_chunk:
        chunks.append({
            "text": "\n".join([t for t, _ in current_chunk]),
            "type": current_type or "NarrativeText"
        })

    return chunks if chunks else [{"text": "No content extracted.", "type": "Unknown"}]


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🎓💬 WU Vienna AI Course Tutor (DeepDoc RAG)")
st.write("Upload course materials and ask questions. All processing is local and private!")

uploaded_files = st.file_uploader(
    "Upload course material(s)",
    type=["pdf", "docx", "pptx", "txt", "xlsx", "xls"],   
    accept_multiple_files=True
)

if st.button("Ingest Documents") and uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(TEMP_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        chunks = extract_chunks_deepdoc(file_path, file.name)
        if not chunks:
            st.warning(f"⚠️ Could not extract text from {file.name}.")
            continue

        for idx, chunk in enumerate(chunks):
            doc_id = f"{file.name}_chunk{idx+1}"
            metadata = {
                "source": file.name,
                "chunk": idx+1,
                "type": chunk["type"]   # <-- NEW: store element type
            }
            collection.add(documents=[chunk["text"]], metadatas=[metadata], ids=[doc_id])

    st.success(f"✅ Indexed {len(uploaded_files)} document(s) into the database.")

# ── Query ─────────────────────────────────────────────────────────────────────

query = st.text_input("Ask a question about your course:")

if st.button("Get Answer") and query:
    # Filter out Tables for prose questions — or flip to {"type": "Table"} for data questions
    results = collection.query(
        query_texts=[query],
        n_results=3,
        where={"type": {"$in": ["NarrativeText", "Title", "ListItem"]}}  # <-- NEW: metadata filter
    )
    if results.get("documents"):
        top_chunks = results["documents"][0]
        context = "\n".join(top_chunks)
    else:
        context = ""

    ollama_url = "http://localhost:11434/api/generate"
    model = "tinyllama"

    prompt_text = (
        f"You are a helpful academic tutor for WU Vienna students. "
        f"Use the following course material excerpts to answer the question.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )

    payload = {"model": model, "prompt": prompt_text, "stream": False}

    try:
        res = requests.post(ollama_url, json=payload, timeout=1200)
        answer = res.json().get("response", "").strip() if res.status_code == 200 else f"Error: {res.status_code}"
    except Exception as e:
        answer = f"Error communicating with Ollama: {e}"

    st.subheader("Answer:")
    st.write(answer)

    with st.expander("Show retrieved context"):
        for chunk in top_chunks:
            st.write(f"- {chunk[:1000]}{'...' if len(chunk) > 1000 else ''}")