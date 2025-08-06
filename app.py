import streamlit as st
import os
import shutil
import traceback
import torch
import re
import sqlite3
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from langchain.docstore.document import Document

# Document loaders
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
)

# ------------------ Persistence Config ------------------
DB_FILE = "docqa.sqlite"
FAISS_INDEX_DIR = "faiss_index"
FAISS_INDEX_FILE = os.path.join(FAISS_INDEX_DIR, "index")

# ------------------ Helper function for short answer extraction ------------------
def extract_short_answer(query, text):
    if any(kw in query.lower() for kw in ["project number", "number", "no.", "id"]):
        match = re.search(r'\d+', text)
        if match:
            return match.group(0)
    return text.strip()

# ------------------ SQLite persistence functions ------------------
def get_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    return conn

def create_tables():
    with get_db() as conn:
        conn.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            text_chunk TEXT
        )
        ''')
        conn.commit()

def save_document_chunk(filename, text_chunk):
    with get_db() as conn:
        conn.execute('INSERT INTO documents (filename, text_chunk) VALUES (?,?)', (filename, text_chunk))
        conn.commit()

def get_all_document_chunks():
    with get_db() as conn:
        rows = conn.execute('SELECT filename, text_chunk FROM documents').fetchall()
    return rows

def clear_documents():
    with get_db() as conn:
        conn.execute('DELETE FROM documents')
        conn.commit()

# ------------------ Document processing ------------------
def process_documents(files):
    all_texts = []
    os.makedirs("temp", exist_ok=True)
    for file in files:
        file_ext = file.name.split(".")[-1].lower()
        file_path = os.path.join("temp", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        try:
            if file_ext == "txt":
                loader = TextLoader(file_path, encoding="utf-8")
            elif file_ext == "pdf":
                loader = PyPDFLoader(file_path)
            elif file_ext == "docx":
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_ext == "pptx":
                loader = UnstructuredPowerPointLoader(file_path)
            else:
                st.warning(f"Unsupported file type: {file.name}")
                continue
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
            texts = splitter.split_documents(docs)
            st.write(f"Loaded and split '{file.name}' into {len(texts)} chunks.")
            # Save chunks to DB with filename metadata
            for t in texts:
                save_document_chunk(file.name, t.page_content)
            all_texts.extend(texts)
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")
            st.text(traceback.format_exc())
    return all_texts

# ------------------ Upload files info tracking ------------------
def load_uploaded_files_info():
    """Load all unique filenames from DB to display."""
    doc_rows = get_all_document_chunks()
    files_seen = {}
    for fn, _ in doc_rows:
        files_seen[fn] = True
    if "uploaded_files_info" not in st.session_state:
        st.session_state.uploaded_files_info = []
    current_files = {f["name"] for f in st.session_state.uploaded_files_info}
    for fn in files_seen.keys():
        if fn not in current_files:
            fpath = os.path.join("temp", fn)
            st.session_state.uploaded_files_info.append({"name": fn, "path": fpath, "type": fn.split('.')[-1].lower()})

# ------------------ Setup ------------------
load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.set_page_config(page_title="📚 Ask Questions from Docs", layout="centered")
st.title("🤖 Ask Questions from Your Documents")

# Sidebar - Upload files
st.sidebar.title("📂 Upload Your Files")

if st.sidebar.button("🚨 Delete all documents and reset DB"):
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    if os.path.exists(FAISS_INDEX_DIR):
        shutil.rmtree(FAISS_INDEX_DIR)
    if "vectorstore" in st.session_state:
        st.session_state.vectorstore = None
    if "uploaded_files_info" in st.session_state:
        st.session_state.uploaded_files_info = []
    st.sidebar.success("All data deleted. Please refresh the page.")

uploaded_files = st.sidebar.file_uploader(
    "Upload .txt, .pdf, .docx, or .pptx files",
    type=["txt", "pdf", "docx", "pptx"],
    accept_multiple_files=True,
)

model_choice = st.sidebar.selectbox(
    "🧠 Choose a Hugging Face model",
    [
        "google/flan-t5-base",
        "google/flan-t5-large",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "deepset/roberta-base-squad2",
        "distilbert-base-cased-distilled-squad"
    ]
)

device = "cuda" if torch.cuda.is_available() else "cpu"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)

# Initialize DB and vectorstore
create_tables()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "uploaded_files_info" not in st.session_state:
    st.session_state.uploaded_files_info = []

# Load or create vectorstore from DB + saved FAISS index
def load_vectorstore():
    if os.path.exists(FAISS_INDEX_DIR) and os.path.exists(FAISS_INDEX_FILE + ".index"):
        try:
            vs = FAISS.load_local(FAISS_INDEX_DIR, embeddings)
            st.session_state.vectorstore = vs
            st.info("Loaded existing vectorstore index from disk.")
        except Exception as e:
            st.warning(f"Error loading saved FAISS index: {e}")
            st.session_state.vectorstore = None
    else:
        doc_rows = get_all_document_chunks()
        if doc_rows:
            texts = [Document(page_content=row[1], metadata={"source": row[0]}) for row in doc_rows]
            try:
                vs = FAISS.from_documents(texts, embeddings)
                os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
                vs.save_local(FAISS_INDEX_DIR)
                st.session_state.vectorstore = vs
                st.info("Built vectorstore from DB documents and saved index.")
            except Exception as e:
                st.error(f"Error creating vectorstore from DB: {e}")
                st.session_state.vectorstore = None
        else:
            st.session_state.vectorstore = None

load_vectorstore()
load_uploaded_files_info()

# ----------------- File upload & indexing -----------------
if uploaded_files:
    with st.spinner("🔄 Processing files and updating index..."):
        clear_documents()
        texts = process_documents(uploaded_files)
        if not texts:
            st.error("⚠️ No valid documents were processed from upload.")
        else:
            try:
                doc_rows = get_all_document_chunks()
                texts = [Document(page_content=row[1], metadata={"source": row[0]}) for row in doc_rows]
                vs = FAISS.from_documents(texts, embeddings)
                os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
                vs.save_local(FAISS_INDEX_DIR)
                st.session_state.vectorstore = vs
                st.success("✅ Documents processed, vectorstore created and persisted!")
                load_uploaded_files_info()
            except Exception as e:
                st.error(f"Error creating vectorstore: {e}")
                st.text(traceback.format_exc())

# ---------------- Sidebar: Display uploaded documents and previews ----------------
st.sidebar.markdown("### 📂 Uploaded Documents")

if st.session_state.uploaded_files_info:
    for f in st.session_state.uploaded_files_info:
        st.sidebar.markdown(f"**{f['name']}**")
        if f["type"] == "pdf":
            try:
                with open(f["path"], "rb") as pdf_file:
                    PDFbyte = pdf_file.read()
                st.sidebar.download_button(
                    label=f"Download {f['name']}",
                    data=PDFbyte,
                    file_name=f['name'],
                    mime='application/pdf'
                )
                st.sidebar.info("Click download to open PDF locally (Streamlit cannot show multi-page PDFs inline)")
            except Exception as e:
                st.sidebar.error(f"Error loading PDF preview: {e}")
        elif f["type"] in ["txt", "docx", "pptx"]:
            try:
                with open(f["path"], "r", encoding="utf-8", errors="ignore") as fpeek:
                    preview = fpeek.read(500)
                st.sidebar.text_area(f"Preview of {f['name']}", preview, height=150)
            except Exception:
                st.sidebar.write(f"Preview unavailable for {f['name']}")
        else:
            st.sidebar.write(f"No preview available for {f['name']}")
else:
    st.sidebar.write("No documents uploaded yet.")

# ------------------ Question Answering Logic ------------------
query = st.text_input("💬 Ask a question based on the uploaded documents:")

if query and st.session_state.vectorstore:
    st.subheader("Question")
    st.markdown(f"**{query}**")
    with st.spinner("💡 Thinking..."):
        try:
            docs = st.session_state.vectorstore.similarity_search(query, k=5)
            if not docs:
                st.warning("No relevant content found.")
                st.markdown("**Answer:** Sorry, there is no such content in the provided documents.")
            else:
                if model_choice in ["deepset/roberta-base-squad2", "distilbert-base-cased-distilled-squad"]:
                    qa_pipeline = pipeline(
                        "question-answering",
                        model=model_choice,
                        tokenizer=model_choice,
                        device=0 if torch.cuda.is_available() else -1,
                    )
                    best_answer = ""
                    best_score = 0
                    best_source = None
                    for doc in docs:
                        result = qa_pipeline(question=query, context=doc.page_content)
                        current_answer = result.get("answer", "").strip()
                        if result.get("score", 0) > best_score and current_answer and current_answer.lower() not in ["", "no answer", "unknown"]:
                            best_answer = current_answer
                            best_score = result["score"]
                            best_source = doc.metadata.get("source", "Unknown")
                    if not best_answer or best_score < 0.3:
                        st.markdown("**Answer:** Sorry, there is no such content in the provided documents.")
                    else:
                        short_answer = extract_short_answer(query, best_answer)
                        st.markdown(f"**Answer:** {short_answer}")
                        st.markdown(f"**Source:** {best_source or 'Unknown'}")
                else:
                    context = "\n---\n".join(doc.page_content for doc in docs[:3] if len(doc.page_content) > 50)
                    prompt = (
                        f"The following excerpts come from uploaded documents. Answer the user's question truthfully using ONLY the information in the context. "
                        f"If not found, reply exactly: 'Sorry, there is no such content in the provided documents.'\n\n"
                        f"Context:\n{context}\n\n"
                        f"Question: {query}\nAnswer:"
                    )
                    tokenizer = AutoTokenizer.from_pretrained(model_choice)
                    if "flan" in model_choice:
                        model = AutoModelForSeq2SeqLM.from_pretrained(model_choice, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
                        pipe_type = "text2text-generation"
                    else:
                        model = AutoModelForCausalLM.from_pretrained(model_choice, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
                        pipe_type = "text-generation"
                    model.to(device)
                    local_pipeline = pipeline(
                        pipe_type,
                        model=model,
                        tokenizer=tokenizer,
                        max_length=1024,
                        temperature=0.3,
                        device=0 if torch.cuda.is_available() else -1,
                    )
                    output = local_pipeline(prompt)[0]
                    generated = output.get("generated_text", "") if pipe_type == "text2text-generation" else output.get("generated_text", "")
                    answer = generated.split("Answer:")[-1].strip() if "Answer:" in generated else generated.strip()
                    short_answer = extract_short_answer(query, answer)
                    if len(short_answer) < 2 or "no such content" in short_answer.lower() or "i do not know" in short_answer.lower():
                        short_answer = "Sorry, there is no such content in the provided documents."
                    st.markdown(f"**Answer:** {short_answer}")
                    for i, doc in enumerate(docs[:3], 1):
                        st.markdown(f"**Source {i}:** {doc.metadata.get('source', 'Unknown')}")
        except Exception as e:
            st.error(f"❌ Error generating answer:\n{str(e)}")
            st.text(traceback.format_exc())

elif query and not st.session_state.vectorstore:
    st.warning("📁 Please upload and process documents first.")

# ------------------ Cleanup temporary files ------------------
if os.path.exists("temp"):
    shutil.rmtree("temp")
