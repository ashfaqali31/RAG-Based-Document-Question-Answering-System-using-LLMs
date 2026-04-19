from langchain_text_splitters import TokenTextSplitter

def split_docs(documents):
    splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

