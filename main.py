from loader import load_pdf
from splitter import split_docs
from embeddings import get_embeddings
from vectorstore import build_faiss

path = "data/ISLP.pdf"

documents = load_pdf(path)
docs = split_docs(documents)

embeddings = get_embeddings()
vectorstore = build_faiss(docs, embeddings)

print("RAG pipeline ready!")

#retrieve relevant documents based on a query
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

#Add LLM

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=pipe)

#Prompt Template
from langchain.prompts import PromptTemplate

prompt_template = """
Answer the question using only the context below.

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

#connect LLM to retriever (RAG Chain)
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

#query the chain
query = "what is statistical learning?"

result = qa_chain.invoke({"query": query})

print("\nANSWER:\n")
print(result["result"])