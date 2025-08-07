# 🤖 Ask Questions from Your Documents

A Streamlit-based AI app that lets you upload `.pdf`, `.docx`, `.pptx`, and `.txt` files and **ask questions** about the content. Built using **LangChain**, **FAISS**, **HuggingFace Transformers**, and **Streamlit**.

🔗 **Live Demo**: [Click here to try the app](https://chat-bot-for-our-documents-acsvjmd38uqdnue9ztp7rd.streamlit.app/)
---

## 🚀 Features

- 📂 Upload and parse `.pdf`, `.docx`, `.pptx`, `.txt`
- 🧠 Choose from multiple Hugging Face QA and generation models
- 💾 Stores chunks in SQLite and indexes them with FAISS
- ❓ Ask questions and get AI-generated answers with source info
- 🧹 One-click cleanup of documents and indexes

---

## 🔧 Installation

```bash
# Clone the repository
git clone (https://github.com/moni-sm/chat-bot-for-our-Documents.git)
cd chatbot-docqa

# Create a virtual environment (conda or venv)
conda create --name chatbot311 python=3.11
conda activate chatbot311

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```
## 🧠 Supported Hugging Face Models

You can select from the following models (CPU/GPU support varies):

- `google/flan-t5-base` ✅ (CPU-friendly)
- `google/flan-t5-large`
- `mistralai/Mistral-7B-Instruct-v0.1` ⚠️ (GPU recommended)
- `deepset/roberta-base-squad2`
- `distilbert-base-cased-distilled-squad`

⚠️ Heavy models like Mistral require a GPU. For CPU environments, use smaller models like `flan-t5-base` or `distilbert-base-cased-distilled-squad`.

## 🧠 Supported Hugging Face Models

You can select from the following models (CPU/GPU support varies):

- `google/flan-t5-base` ✅ (CPU-friendly)
- `google/flan-t5-large`
- `mistralai/Mistral-7B-Instruct-v0.1` ⚠️ (GPU recommended)
- `deepset/roberta-base-squad2`
- `distilbert-base-cased-distilled-squad`

⚠️ Heavy models like Mistral require a GPU. For CPU environments, use smaller models like `flan-t5-base` or `distilbert-base-cased-distilled-squad`.

## 📁 Supported File Types

This app supports the following document formats for Q&A:

- `.pdf` – via `PyPDFLoader`
- `.docx` – via `UnstructuredWordDocumentLoader`
- `.pptx` – via `UnstructuredPowerPointLoader`
- `.txt` – via `TextLoader`

## 🧰 How It Works

1. **Upload documents** via the Streamlit interface.
2. Documents are **split into chunks** using `RecursiveCharacterTextSplitter`.
3. Chunks are **stored in SQLite** and **indexed using FAISS**.
4. When a user asks a question:
   - The app performs a **similarity search** on the chunks.
   - The most relevant chunks are passed to the **selected HuggingFace model**.
5. The model **generates an answer** based on retrieved context.
6. The source documents and chunks used are **displayed with the answer** for transparency.


## 🧼 Reset / Clean All Data

Use the sidebar’s **"🚨 Delete all documents and reset DB"** button to:

- 🗑️ Delete the `docqa.sqlite` database
- 🗂️ Clear the FAISS index directory
- 🧹 Remove all uploaded files from the `temp/` folder

This ensures a clean slate for uploading new documents and starting fresh.

## 🔒 .gitignore

Make sure to include a `.gitignore` file in your project to avoid committing unnecessary files:

pycache/
*.pyc
*.sqlite
faiss_index/
temp/
.env

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🙋‍♀️ Built by Monika SM

A beginner-friendly AI-powered document Q&A app to help users query their own files using NLP and open-source models.

👉 [Follow me on LinkedIn](https://www.linkedin.com/in/monika-sm/) 🌟  
