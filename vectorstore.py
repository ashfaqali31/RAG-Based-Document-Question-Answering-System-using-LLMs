from langchain_community.vectorstores import FAISS

def build_faiss(docs, embeddings):
    return FAISS.from_documents(docs, embeddings)