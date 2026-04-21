# 📄 RAG-Based Document Question Answering System

An end-to-end **Retrieval-Augmented Generation (RAG)** application that allows users to upload a PDF and ask questions based on its content.

Built using **LangChain, Hugging Face Transformers, FAISS, and Streamlit**.

---

## 🚀 Features

* 📄 Upload any PDF document
* 🔍 Intelligent document chunking & embedding
* 🧠 Semantic search using FAISS
* 🤖 Local LLM (TinyLlama / FLAN-T5) for answering
* 💬 Natural language question answering
* 📚 Context-aware responses from the document

---

## 🏗️ Tech Stack

* **Python 3.11**
* **LangChain**
* **FAISS**
* **Hugging Face Transformers**
* **Sentence Transformers**
* **Streamlit**
* **PyTorch**

---

## 📂 Project Structure

```
.
├── app.py               # Streamlit UI
├── main.py              # Core RAG pipeline (CLI)
├── loader.py            # Load PDF
├── splitter.py          # Split documents into chunks
├── embeddings.py        # Generate embeddings
├── vectorstore.py       # FAISS vector database
├── data/
│   └── sample.pdf       # Sample PDF
└── README.md
```

---

## ⚙️ Installation

### 1. Create environment (Python 3.11 recommended)

```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

---

### 2. Install dependencies

```bash
pip install langchain==0.1.20
pip install langchain-community
pip install langchain-core
pip install langchain-text-splitters
pip install transformers==4.41.0
pip install sentence-transformers
pip install faiss-cpu
pip install sentencepiece
pip install python-dotenv
pip install streamlit
pip install torch
```

---

## ▶️ Run the App

```bash
py -3.11 -m streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## 🧠 How It Works

1. **PDF Upload**

   * User uploads a document via Streamlit

2. **Text Processing**

   * PDF is loaded and split into smaller chunks

3. **Embeddings**

   * Each chunk is converted into vector embeddings

4. **Vector Store**

   * Stored in FAISS for efficient similarity search

5. **Retrieval**

   * Top relevant chunks are retrieved based on query

6. **LLM Generation**

   * Local LLM generates answer using retrieved context

---

## 💡 Example Query

> *"What is statistical learning?"*

### Output:

> Statistical learning refers to a set of tools for understanding data. It involves building models to predict or infer outcomes and can be broadly classified into supervised and unsupervised learning.

---

## ⚠️ Notes

* First run may download models from Hugging Face
* Large models require sufficient RAM/VRAM
* TinyLlama is used for faster local inference

---

## 🔮 Future Improvements

* 💬 Chat history (conversation memory)
* 📚 Source highlighting in UI
* ⚡ Model caching for faster performance
* ☁️ Deployment (Hugging Face Spaces / Render)
* 🧠 Better LLM (Mistral / Phi-3)

---

## 👨‍💻 Author

**Ashfaq Ali**
Data Science | AI | LLM Applications

---

## ⭐ If you like this project

Give it a star on GitHub ⭐ and feel free to contribute!
