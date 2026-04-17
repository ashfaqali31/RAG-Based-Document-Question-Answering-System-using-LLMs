from langchain_community.document_loaders import PyPDFLoader

def load_pdf():
    loader = PyPDFLoader(r"C:\Users\ashua\Desktop\LLM-RAG-Project\RAG-Based-Document-Question-Answering-System-using-LLMs\data\ISLP.pdf")
    pages = loader.load()
    return pages

#load the PDF document
documents = load_pdf()

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Split the loaded PDF into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

print(f"Number of chunks: {len(docs)}")