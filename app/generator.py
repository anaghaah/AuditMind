import os
import requests
import streamlit as st
from app.retriever import get_retriever

SYSTEM_PROMPT = """
You are AuditMind AI, an enterprise-grade financial auditing assistant specialized in SEC 10-K filings.
Analyze the provided context documents carefully and answer the user's inquiry accurately.

STRICT AUDIT RULES:
1. Grounding: Rely ONLY on the provided Context. Do NOT use outside general knowledge or hallucinate.
2. Mandatory Company Check: If the user's question asks for financial metrics (such as revenue, net income, cash flow, expenses) WITHOUT explicitly naming the target company (e.g., Apple, Microsoft, NVIDIA), do NOT assume or answer from the context. You MUST immediately respond: "Which company's filings would you like me to audit? (e.g., Apple, Microsoft, NVIDIA)".
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

    api_key = None
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return {
            "answer": "Error: GEMINI_API_KEY not found in Streamlit Secrets or environment variables.",
            "sources": []
        }

    api_key = str(api_key).strip().strip('"').strip("'")
    prompt_content = f"{SYSTEM_PROMPT}\n\nContext:\n{context_text}\n\nQuestion:\n{query}\n\nAudit Answer:"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt_content}]
            }
        ]
    }

    res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)

    if res.status_code == 200:
        data = res.json()
        answer_text = data["candidates"][0]["content"]["parts"][0]["text"]
    else:
        answer_text = f"API Error: {res.text}"

    # Filter out sources that weren't used by the LLM in its response
    used_sources = []
    for doc in relevant_docs:
        source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
        page_num = doc.metadata.get("page", 0) + 1
        
        if source_name in answer_text:
            used_sources.append(f"Document Name: {source_name} | Page Number: {page_num}")

    # Fallback to all retrieved documents if no explicit file name was tagged
    final_sources = list(set(used_sources)) if used_sources else list(set([
        f"Document Name: {os.path.basename(d.metadata.get('source', 'Unknown'))} | Page Number: {d.metadata.get('page', 0) + 1}"
        for d in relevant_docs
    ]))

    return {
        "answer": answer_text,
        "sources": final_sources
    }