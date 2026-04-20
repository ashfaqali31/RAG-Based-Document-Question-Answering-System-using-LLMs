from loader import load_pdf
from splitter import split_docs
from embeddings import get_embeddings
from vectorstore import build_faiss

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ---------------------------
# 1. Load PDF
# ---------------------------
path = "C:\\Users\\ashua\\Desktop\\LLM-RAG-Project\\RAG-Based-Document-Question-Answering-System-using-LLMs\\data\\ISLP.pdf"
documents = load_pdf(path)

# ---------------------------
# 2. Split into chunks
# ---------------------------
docs = split_docs(documents)

# ---------------------------
# 3. Embeddings + Vector DB
# ---------------------------
embeddings = get_embeddings()
vectorstore = build_faiss(docs, embeddings)

print("✅ RAG pipeline ready!")

# ---------------------------
# 4. Retriever
# ---------------------------
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# ---------------------------
# 5. Prompt Template (IMPORTANT FIX)
# ---------------------------
prompt = PromptTemplate(
    template="""
You are a helpful AI assistant.

Use ONLY the context below to answer the question.
Give a detailed explanation (at least 3-4 sentences).
Do not give short answers.

Context:
{context}

Question:
{question}

Detailed Answer:
""",
    input_variables=["context", "question"]
)

# ---------------------------
# 6. LLM and Pipeline
# ---------------------------
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True
)

llm = HuggingFacePipeline(pipeline=pipe)

# ---------------------------
# 7. RAG Chain
# ---------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# ---------------------------
# 8. Query
# ---------------------------
query = "What is statistical learning?"

result = qa_chain.invoke({"query": query})

print("\nANSWER:\n")
print(result["result"])