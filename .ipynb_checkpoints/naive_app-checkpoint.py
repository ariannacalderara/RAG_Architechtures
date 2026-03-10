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

TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

CHROMA_DIR = "chroma_data"
client = chromadb.PersistentClient(path=CHROMA_DIR)
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name="course_docs", embedding_function=embed_fn)

# ── Extractors ────────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_path):
    chunks = []
    doc = fitz.open(file_path)
    for page in doc:
        page_text = page.get_text().strip()
        if page_text:
            chunks.append(page_text)
        else:
            pix = page.get_pixmap(dpi=250)
            img = Image.open(io.BytesIO(pix.pil_tobytes()))
            ocr_text = pytesseract.image_to_string(img).strip()
            if ocr_text:
                chunks.append(ocr_text)
    doc.close()
    return chunks

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    chunks = []
    current_chunk = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        # Start a new chunk on headings
        if para.style.name.startswith("Heading") and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = [text]
        else:
            current_chunk.append(text)
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks if chunks else ["No text found in document."]

def extract_text_from_pptx(file_path):
    prs = Presentation(file_path)
    chunks = []
    for i, slide in enumerate(prs.slides):
        slide_text = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                slide_text.append(shape.text.strip())
        if slide_text:
            chunks.append(f"[Slide {i+1}]\n" + "\n".join(slide_text))
    return chunks if chunks else ["No text found in presentation."]

def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    # Split into ~500 char chunks
    chunk_size = 500
    return [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

def extract_text_from_excel(file_path):
    chunks = []
    xl = pd.ExcelFile(file_path)
    for sheet_name in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet_name)
        df = df.dropna(how="all").fillna("")
        #Each sheet becomes a chunk, serialized as readable text
        sheet_text = f"[Sheet: {sheet_name}]\n"
        sheet_text += df.to_string(index=False)
        chunks.append(sheet_text)
    return chunks if chunks else ["No data found in spreadsheet."]


def extract_chunks(file_path, file_name):
    ext = os.path.splitext(file_name)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".pptx":
        return extract_text_from_pptx(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    elif ext in [".xlsx", ".xls"]:       # <-- add this
        return extract_text_from_excel(file_path)
    else:
        return []


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🎓💬 WU Vienna AI Course Tutor (Naive RAG)")
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

        chunks = extract_chunks(file_path, file.name)
        if not chunks:
            st.warning(f"⚠️ Could not extract text from {file.name} — unsupported format.")
            continue

        for idx, chunk in enumerate(chunks):
            doc_id = f"{file.name}_chunk{idx+1}"
            metadata = {"source": file.name, "chunk": idx+1}
            collection.add(documents=[chunk], metadatas=[metadata], ids=[doc_id])

    st.success(f"✅ Indexed {len(uploaded_files)} document(s) into the database.")

# ── Query ─────────────────────────────────────────────────────────────────────

query = st.text_input("Ask a question about your course:")

if st.button("Get Answer") and query:
    results = collection.query(query_texts=[query], n_results=3)
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