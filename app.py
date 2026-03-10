import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import requests
import fitz
import pytesseract
from PIL import Image
import io

# for uploading of files using Windows
import os
TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)  # ensure folder exists


# Initialize ChromaDB client and collection (persistent)
CHROMA_DIR = "chroma_data"
client = chromadb.PersistentClient(path=CHROMA_DIR)
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name="financial_docs", embedding_function=embed_fn)
# Function to extract text from PDF pages 
def extract_text_from_pdf(file_path):
    text_chunks = []
    doc = fitz.open(file_path)
    for page in doc:
        page_text = page.get_text().strip()
        if page_text:
            text_chunks.append(page_text)
        else:
            pix = page.get_pixmap(dpi=250)
            img = Image.open(io.BytesIO(pix.pil_tobytes()))
            ocr_text = pytesseract.image_to_string(img).strip()
            if ocr_text:
                text_chunks.append(ocr_text)
    doc.close()
    return text_chunks

st.title("📄💬 Chat with Your Financial Documents (Local RAG)")
st.write("Upload your bank statements or bills as PDFs, and ask questions. All processing is local and private!")
# File uploader (multiple files allowed)
uploaded_files = st.file_uploader("Upload PDF document(s)", type=["pdf"], accept_multiple_files=True)
# Button to add documents to the vector store
if st.button("Ingest Documents") and uploaded_files:
    for file in uploaded_files:
        # Save file to disk (for Linux)
        # file_path = f"/tmp/{file.name}"
        # Save file to disk (for Windows)
        file_path = os.path.join(TEMP_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        # Extract and index text
        chunks = extract_text_from_pdf(file_path)
        for idx, chunk in enumerate(chunks):
            doc_id = f"{file.name}_page{idx+1}"
            metadata = {"source": file.name, "page": idx+1}
            collection.add(documents=[chunk], metadatas=[metadata], ids=[doc_id])
    st.success(f"Indexed {len(uploaded_files)} document(s) into the database.")
# Question input
query = st.text_input("Ask a question about your documents:")
if st.button("Get Answer") and query:
    # Retrieve relevant context
    results = collection.query(query_texts=[query], n_results=3)
    if results.get("documents"):
        top_chunks = results["documents"][0]
        context = "\n".join(top_chunks)
    else:
        context = ""
    # Query LLM via Ollama
    ollama_url = "http://localhost:11434/api/generate"
    # Query LLM via Mistral:phi3:llama2:7b
    model = "qwen3.5:0.8b"
    # model = "mistral:7b"
    prompt_text = (f"Use the following document excerpts to answer the question.\n\n"
                   f"Context:\n{context}\n\nQuestion: {query}\nAnswer:")
    payload = {"model": model, "prompt": prompt_text, "stream": False}

    try:
        res = requests.post(ollama_url, json=payload, timeout=1200)
        if res.status_code == 200:
            answer = res.json().get("response", "").strip()
        else:
            answer = f"Error: Ollama returned status {res.status_code}"
    except Exception as e:
        answer = f"Error communicating with Ollama: {e}"
    # Display the answer
    st.subheader("Answer:")
    st.write(answer)
    # Optionally, show the context that was used (for transparency or debugging)
    with st.expander("Show retrieved context"):
        for chunk in top_chunks:
            st.write(f"- {chunk[:1000]}{'...' if len(chunk)>1000 else ''}")