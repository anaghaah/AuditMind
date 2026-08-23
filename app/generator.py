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

    # API key
    api_key = None
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return {
            "answer": "Error: GEMINI_API_KEY not found.",
            "sources": []
        }

    api_key = str(api_key).strip().strip('"').strip("'")
    prompt_content = f"{SYSTEM_PROMPT}\n\nContext:\n{context_text}\n\nQuestion:\n{query}\n\nAudit Answer:"

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt_content}]
            }
        ]
    }

    # Step A: Dynamically list supported models for your API key
    list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    target_model = None

    try:
        res = requests.get(list_url, timeout=10)
        if res.status_code == 200:
            models_data = res.json().get("models", [])
            # Prioritize flash or any generateContent capable model
            for m in models_data:
                methods = m.get("supportedGenerationMethods", [])
                name = m.get("name", "")
                if "generateContent" in methods:
                    if "flash" in name:
                        target_model = name
                        break
                    elif not target_model:
                        target_model = name
    except Exception:
        pass

    # Fallback to gemini-2.5-flash or gemini-2.0-flash if list failed
    if not target_model:
        target_model = "models/gemini-2.5-flash"

    # Step B: Call the active model
    generate_url = f"https://generativelanguage.googleapis.com/v1beta/{target_model}:generateContent?key={api_key}"
    res = requests.post(generate_url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)

    if res.status_code == 200:
        data = res.json()
        answer_text = data["candidates"][0]["content"]["parts"][0]["text"]
    else:
        answer_text = f"API Error ({res.status_code}): {res.text}"

    sources = [
        f"Document Name: {os.path.basename(doc.metadata.get('source', 'Unknown'))} | Page Number: {doc.metadata.get('page', 0) + 1}"
        for doc in relevant_docs
    ]

    return {
        "answer": answer_text,
        "sources": list(set(sources))
    }