import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

def run_ingestion():
    print("🚀 Ingesting all 10-K filings from data folder...")
    
    loader = PyPDFDirectoryLoader(DATA_DIR)
    documents = loader.load()
    print(f"📄 Loaded {len(documents)} total pages across all files.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"🧩 Created {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print("✅ All filings vectorized and stored in ChromaDB successfully!")

if __name__ == "__main__":
    run_ingestion()