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
from datetime import datetime
from unstructured.partition.auto import partition
from unstructured.documents.elements import Table, Title, NarrativeText, ListItem

# ── ReportLab imports for PDF export ──────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table as RLTable, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)

TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

CHROMA_DIR = "chroma_data"
client = chromadb.PersistentClient(path=CHROMA_DIR)
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name="course_docs", embedding_function=embed_fn)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "tinyllama"


# ── Test questions ─────────────────────────────────────────────────────────────
TEST_QUESTIONS = [
    # ── Q01-02: Chunking — DOCX argument spans multiple sections ─────────────
    {
        "id": "Q01",
        "question": "What are the disadvantages of the Service Center model, and how do they relate to the advantages listed for the same model earlier in the document?",
        "category": "Chunking"
    },
    {
        "id": "Q02",
        "question": "The document describes four IT department positioning models. What overall conclusion can be drawn about when to use each model based on the full document?",
        "category": "Chunking"
    },
    # ── Q03-05: Table parsing — Excel with merged headers ────────────────────
    {
        "id": "Q03",
        "question": "What is the definition of the Profit Center model as it appears in the third row of the table in the spreadsheet?",
        "category": "Table Parsing"
    },
    {
        "id": "Q04",
        "question": "According to the spreadsheet, what are the advantages of the Investment Center model?",
        "category": "Table Parsing"
    },
    {
        "id": "Q05",
        "question": "What do the four column headers of the main table in the spreadsheet represent?",
        "category": "Table Parsing"
    },
    # ── Q06-07: Image blind spot — slides with no extractable text ───────────
    {
        "id": "Q06",
        "question": "What does the Strategic Alignment Model diagram in the slides illustrate?",
        "category": "Image Blind Spot"
    },
    {
        "id": "Q07",
        "question": "Describe the visual content of the image-only slide that appears after the Practices of IT Governance slide.",
        "category": "Image Blind Spot"
    },
    # ── Q08-09: Hallucination — facts not present in any document ────────────
    {
        "id": "Q08",
        "question": "What percentage of companies surveyed use the Cost Center model as their primary IT department structure according to the course materials?",
        "category": "Hallucination"
    },
    {
        "id": "Q09",
        "question": "What grade did students receive on average for the IT Governance case study assignment in the previous year?",
        "category": "Hallucination"
    },
    # ── Q10-11: Layout parsing — two-column tables in slides PDF ─────────────
    {
        "id": "Q10",
        "question": "According to the minimum baseline of IT governance practices, what Processes are listed alongside the Structures column?",
        "category": "Layout Parsing"
    },
    {
        "id": "Q11",
        "question": "What are the Structures, Processes and Relational Mechanisms listed in the Van Grembergen IT governance practices framework in the slides?",
        "category": "Layout Parsing"
    },
    # ── Q12: Retrieval miss — short agenda slide, low embedding similarity ────
    {
        "id": "Q12",
        "question": "What four questions are listed on the IT Governance and Control agenda slide at the start of the course?",
        "category": "Retrieval Miss"
    },
]


# ── Extractor ─────────────────────────────────────────────────────────────────

def extract_chunks_deepdoc(file_path, file_name):
    elements = partition(filename=file_path)

    chunks = []
    current_chunk = []
    current_type = None

    for el in elements:
        text = str(el).strip()
        if not text:
            continue
        el_type = type(el).__name__

        if isinstance(el, Title) and current_chunk:
            chunks.append({
                "text": "\n".join([t for t, _ in current_chunk]),
                "type": current_type or "NarrativeText"
            })
            current_chunk = []

        current_chunk.append((text, el_type))
        current_type = el_type

    if current_chunk:
        chunks.append({
            "text": "\n".join([t for t, _ in current_chunk]),
            "type": current_type or "NarrativeText"
        })

    return chunks if chunks else [{"text": "No content extracted.", "type": "Unknown"}]

# ── RAG helpers ───────────────────────────────────────────────────────────────

def retrieve_chunks(question, n=3):
    # Try with type filter first
    try:
        results = collection.query(
            query_texts=[question],
            n_results=n,
            where={"type": {"$in": ["NarrativeText", "Title", "ListItem"]}}
        )
        docs  = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        if docs:
            return docs, metas
    except Exception:
        pass
    # Fallback: no filter
    results = collection.query(query_texts=[question], n_results=n)
    docs  = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return docs, metas

def ask_llm(context, question):
    prompt = (
        "You are a helpful academic tutor for WU Vienna students. "
        "Use the following course material excerpts to answer the question.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    payload = {"model": MODEL, "prompt": prompt, "stream": False}
    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=120)
        return res.json().get("response", "").strip() if res.status_code == 200 else f"Error: {res.status_code}"
    except Exception as e:
        return f"Error communicating with Ollama: {e}"

# ── PDF export ────────────────────────────────────────────────────────────────

def build_failure_pdf(results: list, architecture: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()

    title_style   = ParagraphStyle("CT", parent=styles["Title"],
                                   fontSize=18, spaceAfter=6,
                                   textColor=colors.HexColor("#1a4a3a"))
    subtitle_style = ParagraphStyle("CS", parent=styles["Normal"],
                                    fontSize=11, spaceAfter=4,
                                    textColor=colors.HexColor("#444444"))
    h2_style      = ParagraphStyle("H2", parent=styles["Heading2"],
                                   fontSize=13, spaceBefore=10, spaceAfter=4,
                                   textColor=colors.HexColor("#1a4a3a"))
    label_style   = ParagraphStyle("LB", parent=styles["Normal"],
                                   fontSize=8, textColor=colors.HexColor("#888888"),
                                   spaceAfter=2)
    body_style    = ParagraphStyle("BD", parent=styles["Normal"],
                                   fontSize=10, leading=14, spaceAfter=6)
    chunk_style   = ParagraphStyle("CK", parent=styles["Normal"],
                                   fontSize=9, leading=13,
                                   textColor=colors.HexColor("#333333"),
                                   backColor=colors.HexColor("#f0faf5"),
                                   leftIndent=8, rightIndent=8, spaceAfter=4)
    answer_style  = ParagraphStyle("AN", parent=styles["Normal"],
                                   fontSize=10, leading=14,
                                   textColor=colors.HexColor("#1a4a3a"),
                                   leftIndent=8)

    story = []

    # Cover
    story.append(Spacer(1, 3*cm))
    story.append(Paragraph("Experiment 2: Failure Analysis", title_style))
    story.append(Paragraph("RAG Architecture Diagnosis Worksheet", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=colors.HexColor("#a8d5c2"), spaceAfter=12))
    story.append(Spacer(1, 0.5*cm))

    meta_data = [
        ["Architecture tested:", architecture],
        ["Date:", datetime.now().strftime("%d %B %Y")],
        ["Total cases:", str(len(results))],
    ]
    meta_tbl = RLTable(meta_data, colWidths=[5*cm, 10*cm])
    meta_tbl.setStyle(TableStyle([
        ("FONTSIZE",  (0,0), (-1,-1), 10),
        ("TEXTCOLOR", (0,0), (0,-1),  colors.HexColor("#888888")),
        ("FONTNAME",  (0,0), (0,-1),  "Helvetica"),
        ("FONTNAME",  (1,0), (1,-1),  "Helvetica-Bold"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 1.5*cm))

    # Instructions
    story.append(Paragraph("Instructions", h2_style))
    story.append(Paragraph(
        "For each case below, read the question, the retrieved context chunks, and the LLM answer. "
        "Then fill in the diagnosis box:<br/><br/>"
        "<b>1. Which RAG architecture most likely produced this failure?</b> "
        "Circle one: <i>Naive RAG &nbsp;|&nbsp; DeepDoc RAG &nbsp;|&nbsp; Tablebook RAG</i><br/><br/>"
        "<b>2. What type of failure is this?</b> Circle one: "
        "<b>R</b> Retrieval Miss &nbsp; <b>C</b> Chunking Break &nbsp; "
        "<b>T</b> Table/Format Loss &nbsp; <b>I</b> Image Blind Spot &nbsp; "
        "<b>H</b> Hallucination &nbsp; <b>L</b> Layout Error<br/><br/>"
        "<b>3. Explain in 1–2 sentences why this failure occurred.</b>",
        body_style))
    story.append(PageBreak())

    # One case per page
    for i, r in enumerate(results):
        block = []

        header_data = [[
            Paragraph(f"Case {r['id']}",
                      ParagraphStyle("CN", parent=styles["Normal"],
                                     fontSize=13, textColor=colors.white,
                                     fontName="Helvetica-Bold")),
            Paragraph(r["category"],
                      ParagraphStyle("CC", parent=styles["Normal"],
                                     fontSize=10, textColor=colors.HexColor("#a8d5c2")))
        ]]
        header_tbl = RLTable(header_data, colWidths=[3*cm, 13.5*cm])
        header_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), colors.HexColor("#1a4a3a")),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("LEFTPADDING",   (0,0), (0,0),   10),
            ("LEFTPADDING",   (1,0), (1,0),   6),
            ("TOPPADDING",    (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ]))
        block.append(header_tbl)
        block.append(Spacer(1, 0.3*cm))

        block.append(Paragraph("QUESTION", label_style))
        block.append(Paragraph(r["question"], body_style))
        block.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#a8d5c2"), spaceAfter=8))

        #block.append(Paragraph("RETRIEVED CONTEXT CHUNKS", label_style))
        #for j, chunk in enumerate(r["chunks"], 1):
            #source = r["sources"][j-1] if j-1 < len(r["sources"]) else "unknown"
            #block.append(Paragraph(
                #f"<b>Chunk {j}</b> &nbsp; "
                #f"<font color='#888888' size='8'>[{source}]</font>",
                #ParagraphStyle("CH", parent=styles["Normal"],
                               #fontSize=9, spaceAfter=2)))
            #preview = chunk[:600] + ("…" if len(chunk) > 600 else "")
            #preview = preview.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            #block.append(Paragraph(preview, chunk_style))

        #block.append(HRFlowable(width="100%", thickness=0.5,
                                #color=colors.HexColor("#a8d5c2"), spaceAfter=8))

        block.append(Paragraph("LLM ANSWER", label_style))
        answer_text = r["answer"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        block.append(Paragraph(
            answer_text[:800] + ("…" if len(answer_text) > 800 else ""),
            answer_style))
        block.append(Spacer(1, 0.4*cm))
        block.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor("#1a4a3a"), spaceAfter=8))

        block.append(Paragraph("YOUR DIAGNOSIS",
                               ParagraphStyle("DL", parent=styles["Normal"],
                                              fontSize=9, fontName="Helvetica-Bold",
                                              textColor=colors.HexColor("#1a4a3a"),
                                              spaceAfter=6)))
        diag_data = [
            [Paragraph("<b>Architecture:</b>  Naive RAG &nbsp;&nbsp; | &nbsp;&nbsp; "
                       "DeepDoc RAG &nbsp;&nbsp; | &nbsp;&nbsp; Tablebook RAG",
                       ParagraphStyle("DT", parent=styles["Normal"], fontSize=10))],
            [Paragraph("<b>Failure type:</b>  R &nbsp; C &nbsp; T &nbsp; I &nbsp; H &nbsp; L &nbsp; O",
                       ParagraphStyle("DT2", parent=styles["Normal"], fontSize=10))],
            [Paragraph("<b>Explanation:</b><br/><br/><br/>",
                       ParagraphStyle("DT3", parent=styles["Normal"], fontSize=10))],
        ]
        diag_tbl = RLTable(diag_data, colWidths=[16.5*cm])
        diag_tbl.setStyle(TableStyle([
            ("BOX",           (0,0), (-1,-1), 1,   colors.HexColor("#1a4a3a")),
            ("INNERGRID",     (0,0), (-1,-1), 0.5, colors.HexColor("#a8d5c2")),
            ("BACKGROUND",    (0,0), (-1,-1), colors.HexColor("#f0faf5")),
            ("TOPPADDING",    (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ]))
        block.append(diag_tbl)

        story.append(KeepTogether(block))
        if i < len(results) - 1:
            story.append(PageBreak())

    doc.build(story)
    return buf.getvalue()

# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="WU Vienna AI Course Tutor", page_icon="🎓", layout="wide")
st.title("🎓💬 WU Vienna AI Course Tutor (DeepDoc RAG)")
st.write("Upload course materials and ask questions. All processing is local and private!")

tab_chat, tab_export = st.tabs(["💬 Chat", "📋 Experiment 2 — Failure Export"])

# ── Chat tab ───────────────────────────────────────────────────────────────────
with tab_chat:
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
                collection.add(
                    documents=[chunk["text"]],
                    metadatas={"source": file.name, "chunk": idx+1, "type": chunk["type"]},
                    ids=[f"{file.name}_chunk{idx+1}"]
                )
        st.success(f"✅ Indexed {len(uploaded_files)} document(s) into the database.")

    query = st.text_input("Ask a question about your course:")
    if st.button("Get Answer") and query:
        chunks, metas = retrieve_chunks(query)
        context = "\n".join(chunks)
        answer = ask_llm(context, query)
        st.subheader("Answer:")
        st.write(answer)
        with st.expander("Show retrieved context"):
            for chunk in chunks:
                st.write(f"- {chunk[:1000]}{'...' if len(chunk) > 1000 else ''}")

# ── Export tab ─────────────────────────────────────────────────────────────────
with tab_export:
    st.subheader("Generate Student Worksheet")
    st.write(
        "Runs all 12 test questions against the current DeepDoc collection and exports "
        "a printable PDF worksheet — one failure case per page."
    )

    architecture_label = "DeepDoc RAG"

    if st.button("🚀 Run all test questions & generate PDF"):
        results = []
        progress = st.progress(0)
        status = st.empty()

        for i, q in enumerate(TEST_QUESTIONS):
            status.write(f"Running {q['id']}: {q['question'][:60]}…")
            chunks, metas = retrieve_chunks(q["question"], n=3)
            context = "\n---\n".join(chunks) if chunks else ""
            answer = ask_llm(context, q["question"])
            sources = [m.get("source", "?") for m in metas]
            results.append({
                "id":       q["id"],
                "question": q["question"],
                "category": q["category"],
                "chunks":   chunks,
                "sources":  sources,
                "answer":   answer,
            })
            progress.progress((i + 1) / len(TEST_QUESTIONS))

        status.write("✅ All questions run — building PDF…")

        try:
            pdf_bytes = build_failure_pdf(results, architecture_label)
            filename = (f"failure_worksheet_"
                        f"{architecture_label.lower().replace(' ', '_')}_"
                        f"{datetime.now().strftime('%Y%m%d')}.pdf")
            st.success(f"PDF ready! {len(results)} cases exported.")
            st.download_button(
                label="⬇️ Download Worksheet PDF",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"PDF generation failed: {e}")

    st.markdown("---")
    st.markdown("**Failure category codes:**")
    cols = st.columns(3)
    codes = [
        ("R", "Retrieval Miss",    "Wrong chunks fetched"),
        ("C", "Chunking Break",    "Answer split across chunk boundary"),
        ("T", "Table/Format Loss", "Table structure destroyed in parsing"),
        ("I", "Image Blind Spot",  "Content was a diagram or image"),
        ("H", "Hallucination",     "LLM invented facts not in context"),
        ("L", "Layout Error",      "Multi-column or footnote parsed wrong"),
    ]
    for i, (code, name, desc) in enumerate(codes):
        with cols[i % 3]:
            st.markdown(f"**{code} — {name}**  \n{desc}")