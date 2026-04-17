from langchain_text_splitters import RecursiveCharacterTextSplitter

# Split the loaded PDF into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

print(f"Number of chunks: {len(docs)}")