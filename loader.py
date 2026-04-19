from langchain_community.document_loaders import PyPDFLoader

def load_pdf(path):
    loader = PyPDFLoader(path)
    pages = loader.load()
    return pages

#load the PDF document
documents = load_pdf(
    r"C:\Users\ashua\Desktop\LLM-RAG-Project\RAG-Based-Document-Question-Answering-System-using-LLMs\data\ISLP.pdf"
)