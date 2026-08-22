import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_PATH = "data/sample_report.pdf"
DB_DIR = "app/chroma_db"

def build_vector_store():
    print("⏳ Loading PDF...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ {DATA_PATH} file not found! Please ensure the PDF is placed in the 'data' directory.")

    loader = PyPDFLoader(DATA_PATH)
    raw_docs = loader.load()
    print(f"✅ Loaded {len(raw_docs)} pages.")

    print("⏳ Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_docs)
    print(f"✅ Created {len(chunks)} chunks.")

    print("⏳ Creating embeddings using Free HuggingFace model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print("🎉 Vector store created successfully in app/chroma_db!")
    return vectorstore

if __name__ == "__main__":
    build_vector_store()