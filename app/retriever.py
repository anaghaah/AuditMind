import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DB_DIR = "app/chroma_db"

def get_retriever():
    if not os.path.exists(DB_DIR):
        raise FileNotFoundError("❌ ChromaDB not found! Please run ingestion.py to create the vector store first.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )
    
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def test_search(query: str):
    """Test function to check if the search works correctly"""
    print(f"\n🔍 Searching for: '{query}'\n" + "-"*50)
    retriever = get_retriever()
    results = retriever.invoke(query)
    
    for i, doc in enumerate(results, 1):
        print(f"\n--- [Result {i}] ---")
        print(doc.page_content[:300] + "...") 
    print("\n" + "="*50)

if __name__ == "__main__":
    #Sample test query to check if the search works correctly.
    test_search("What is the total revenue and R&D expenses?")