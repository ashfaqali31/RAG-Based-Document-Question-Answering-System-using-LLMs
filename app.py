import streamlit as st
import os
import warnings

# Optional: clean logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from loader import load_pdf
from splitter import split_docs
from embeddings import get_embeddings
from vectorstore import build_faiss

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="RAG Document QA", page_icon="📄")
st.title("📄 RAG Document Question Answering System")

# ---------------------------
# Upload PDF
# ---------------------------
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:

    # Save file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("File uploaded successfully!")

    # ---------------------------
    # Build RAG (cached for speed)
    # ---------------------------
    @st.cache_resource
    def build_pipeline():

        documents = load_pdf("temp.pdf")
        docs = split_docs(documents)

        embeddings = get_embeddings()
        vectorstore = build_faiss(docs, embeddings)

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # ---------------------------
        # LLM
        # ---------------------------
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto"
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=False,
        )

        return retriever, pipe

    with st.spinner("Processing PDF..."):
        retriever, pipe = build_pipeline()

    st.success("✅ RAG pipeline ready!")

    # ---------------------------
    # Query
    # ---------------------------
    query = st.text_input("Ask a question about the document:")

    if query:
        with st.spinner("Thinking..."):

            # Step 1: Retrieve relevant chunks
            docs = retriever.invoke(query)

            context = "\n\n".join([doc.page_content for doc in docs])

            # Step 2: Clean prompt
            prompt = f"""
You are a helpful AI assistant.

Answer the question using ONLY the context below.
Do not repeat the context.
Give a clear and concise answer in 3-4 sentences.

Context:
{context}

Question:
{query}

Answer:
"""

            # Step 3: Generate response
            output = pipe(prompt)[0]["generated_text"]

            # Step 4: Clean output
            answer = output.split("Answer:")[-1].strip()

        st.subheader("Answer:")
        st.write(answer)

        # ---------------------------
        # Show sources (optional)
        # ---------------------------
        with st.expander("📚 Source chunks"):
            for i, doc in enumerate(docs):
                st.write(f"**Chunk {i+1}:**")
                st.write(doc.page_content[:500])
                st.write("---")

