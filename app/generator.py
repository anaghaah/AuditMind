import os
import time
import requests
import streamlit as st
from app.retriever import get_retriever

SYSTEM_PROMPT = """
You are AuditMind AI, an enterprise-grade financial auditing assistant specialized in SEC 10-K filings.
Analyze the provided context documents and immediate prior conversation turn carefully to answer the user's inquiry accurately.

STRICT AUDIT RULES:
1. Grounding: Rely ONLY on the provided Context and Immediate Prior Turn. Do NOT use outside general knowledge or hallucinate.
2. Disambiguation: If the current query asks for financial metrics without specifying a company, and no company is identifiable from the immediately preceding message, ask: "Which company's filings would you like me to audit? (e.g., Apple, Microsoft, NVIDIA)".
3. Precision: Cite exact fiscal years and monetary figures directly from the source text.
4. Source Attribution: Always specify the source document name and page number for every data point.
"""

def generate_answer(query: str, chat_history: list = None):
    retriever = get_retriever(k=8)
    
    # Extract only the immediate previous turn
    history_context = ""
    last_user_query = ""
    if chat_history and len(chat_history) > 1:
        immediate_prior_turn = chat_history[-2:]
        recent_turns = [f"{'User' if m['role'] == 'user' else 'Auditor'}: {m['content']}" for m in immediate_prior_turn]
        history_context = "\n".join(recent_turns)
        
        for m in reversed(immediate_prior_turn):
            if m["role"] == "user":
                last_user_query = m["content"]
                break

    companies = ["apple", "microsoft", "nvidia"]
    mentioned_companies = [c for c in companies if c in query.lower()]

    relevant_docs = []
    if "compare" in query.lower() or len(mentioned_companies) > 1:
        relevant_docs.extend(retriever.invoke(query))
        for comp in mentioned_companies:
            relevant_docs.extend(retriever.invoke(f"{comp} total revenue net income financial results"))
    else:
        search_query = query
        if last_user_query and len(query.strip().split()) <= 3:
            search_query = f"{last_user_query} {query}"
        relevant_docs = retriever.invoke(search_query)

    seen_chunks = set()
    unique_docs = []
    for doc in relevant_docs:
        identifier = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:60])
        if identifier not in seen_chunks:
            seen_chunks.add(identifier)
            unique_docs.append(doc)

    context_text = ""
    for doc in unique_docs:
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
    
    prompt_content = f"""{SYSTEM_PROMPT}

Immediate Previous Turn:
{history_context if history_context else 'None'}

Retrieved SEC Filings Context:
{context_text}

Current User Query: {query}

Audit Answer:"""

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt_content}]
            }
        ]
    }

    # Model fallback hierarchy in case of 429 quota exhaustion
    model_endpoints = [
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-8b:generateContent"
    ]

    answer_text = None
    last_err_msg = ""

    for endpoint in model_endpoints:
        url = f"{endpoint}?key={api_key}"
        res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
        
        if res.status_code == 200:
            data = res.json()
            answer_text = data["candidates"][0]["content"]["parts"][0]["text"]
            break
        elif res.status_code == 429:
            last_err_msg = "Rate limit reached. Please wait ~15 seconds and try again."
            continue  # Try the next model endpoint immediately
        else:
            last_err_msg = f"API Error ({res.status_code}): {res.text}"

    if not answer_text:
        answer_text = last_err_msg

    used_sources = []
    for doc in unique_docs:
        source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
        page_num = doc.metadata.get("page", 0) + 1
        if source_name in answer_text:
            used_sources.append(f"Document Name: {source_name} | Page Number: {page_num}")

    final_sources = list(set(used_sources)) if used_sources else list(set([
        f"Document Name: {os.path.basename(d.metadata.get('source', 'Unknown'))} | Page Number: {d.metadata.get('page', 0) + 1}"
        for d in unique_docs
    ]))

    return {
        "answer": answer_text,
        "sources": final_sources
    }