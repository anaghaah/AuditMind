import os
import requests
import streamlit as st
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

    prompt_content = f"{SYSTEM_PROMPT}\n\nContext:\n{context_text}\n\nQuestion:\n{query}\n\nAudit Answer:"

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt_content}]
            }
        ],
        "generationConfig": {
            "temperature": 0.0
        }
    }

    # Use Stable v1 API & Active Fallback Endpoints
    models_to_try = [
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-8b:generateContent"
    ]

    answer_text = None
    for endpoint in models_to_try:
        url = f"{endpoint}?key={api_key}"
        res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        if res.status_code == 200:
            data = res.json()
            answer_text = data["candidates"][0]["content"]["parts"][0]["text"]
            break

    if not answer_text:
        answer_text = "Unable to generate audit answer. Please verify your GEMINI_API_KEY."

    sources = [
        f"Document Name: {os.path.basename(doc.metadata.get('source', 'Unknown'))} | Page Number: {doc.metadata.get('page', 0) + 1}"
        for doc in relevant_docs
    ]

    return {
        "answer": answer_text,
        "sources": list(set(sources))
    }