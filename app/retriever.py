import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.ingestion import run_ingestion

DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

def get_retriever():
    # If ChromaDB does not exist (like on Streamlit Cloud), create it automatically
    if not os.path.exists(DB_DIR):
        print("⚡ ChromaDB not found. Running automatic ingestion...")
        run_ingestion()

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})