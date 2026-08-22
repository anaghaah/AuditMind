import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.retriever import get_retriever

load_dotenv()

PROMPT_TEMPLATE = """
You are AuditMind, a professional financial auditor assistant.
Answer the user's question clearly and accurately using ONLY the provided context below.

Rules:
1. Answer the question directly using facts and numbers from the context.
2. At the end of your answer, include a dedicated "Sources & Citations" section listing the Document Name and Page Number(s) referenced.
3. If the answer is not contained in the context, state: "The provided document does not contain this specific information."
4. If the user's question is ambiguous or does not specify a company when multiple companies are available in the context, ask the user to specify which company they are inquiring about.

Context:
{context}

Question:
{question}

Answer:
"""

def generate_answer(query: str):
    retriever = get_retriever()
    retriever.search_kwargs = {"k": 5}
    relevant_docs = retriever.invoke(query)
    
    # Format the context with source and page information
    formatted_context_list = []
    for doc in relevant_docs:
        page_num = doc.metadata.get("page", 0) + 1  #0-based indexing to 1-based
        source_file = os.path.basename(doc.metadata.get("source", "10-K Report"))
        chunk_header = f"[Source: {source_file} | Page: {page_num}]"
        formatted_context_list.append(f"{chunk_header}\n{doc.page_content}")
    
    context_text = "\n\n---\n\n".join(formatted_context_list)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "❌ Missing GEMINI_API_KEY in .env file!"

    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=api_key, temperature=0)
    
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({"context": context_text, "question": query})

if __name__ == "__main__":
    test_query = "What is the total revenue for fiscal year 2025?"
    print(f"\n❓ Question: {test_query}\n")
    print("🤖 AuditMind is auditing...")
    answer = generate_answer(test_query)
    print("\n💡 Response:\n" + answer)