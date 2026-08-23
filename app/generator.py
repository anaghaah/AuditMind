import os
import streamlit as st
import google.generativeai as genai
from app.retriever import get_retriever

SYSTEM_PROMPT = """
You are AuditMind AI, an enterprise-grade financial auditing assistant specialized in SEC 10-K filings.
Analyze the provided context documents carefully and answer the user's inquiry accurately.

STRICT AUDIT RULES:
1. Grounding: Rely ONLY on the provided Context. Do NOT use outside general knowledge or hallucinate.
2. Company Ambiguity: If the user asks a metric question (e.g., revenue, net income, cash flow) WITHOUT specifying a company name, and the context contains multiple entities or is unclear, do NOT assume. Explicitly ask the user: "Which company's filings would you like me to audit?"
3. Precision: Cite exact fiscal years and monetary figures directly from the source text.
4. Source Attribution: Always specify the source document name and page number for every data point.
"""

def generate_answer(query: str):
    retriever = get_retriever()
    relevant_docs = retriever.invoke(query)

    context_text = ""
    for doc in relevant_docs:
        source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
        page_num = doc.metadata.get("page", 0) + 1
        context_text += f"\n[Document: {source_name} | Page: {page_num}]\n{doc.page_content}\n"

    # Direct API key retrieval
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass

    genai.configure(api_key=api_key)
    
    # Try gemini-2.0-flash, fallback to gemini-pro if needed
    try:
        model = genai.GenerativeModel("gemini-2.0-flash", system_instruction=SYSTEM_PROMPT)
        user_message = f"Context:\n{context_text}\n\nQuestion:\n{query}\n\nAudit Answer:"
        response = model.generate_content(user_message)
    except Exception:
        model = genai.GenerativeModel("gemini-pro")
        user_message = f"{SYSTEM_PROMPT}\n\nContext:\n{context_text}\n\nQuestion:\n{query}\n\nAudit Answer:"
        response = model.generate_content(user_message)

    sources = [
        f"Document Name: {os.path.basename(doc.metadata.get('source', 'Unknown'))} | Page Number: {doc.metadata.get('page', 0) + 1}"
        for doc in relevant_docs
    ]

    return {
        "answer": response.text,
        "sources": list(set(sources))
    }