import os
import requests
import streamlit as st
from app.retriever import get_retriever

SYSTEM_PROMPT = """
You are AuditMind AI, an enterprise-grade financial auditing assistant specialized in SEC 10-K filings.
Analyze the provided context documents and prior conversation history carefully to answer the user's inquiry accurately.

STRICT AUDIT RULES:
1. Grounding: Rely ONLY on the provided Context and Conversation History. Do NOT use outside general knowledge or hallucinate.
2. Disambiguation: If the user asks for financial metrics without specifying a company, and no company is identifiable from recent messages, ask: "Which company's filings would you like me to audit? (e.g., Apple, Microsoft, NVIDIA)".
3. Conversation Continuity: If the user provides a follow-up answer or comparison, evaluate it in context with previous messages and provided filings.
4. Precision: Cite exact fiscal years and monetary figures directly from the source text.
5. Source Attribution: Always specify the source document name and page number for every data point.
"""

def generate_answer(query: str, chat_history: list = None):
    retriever = get_retriever(k=8)
    
    # Check for comparative or multi-entity context
    companies = ["apple", "microsoft", "nvidia"]
    mentioned_companies = [c for c in companies if c in query.lower()]

    # If it's a comparison or follow-up, retrieve for individual entities to prevent one dominating the vector search
    relevant_docs = []
    if "compare" in query.lower() or len(mentioned_companies) > 1:
        # Retrieve specifically for the new query
        relevant_docs.extend(retriever.invoke(query))
        for comp in mentioned_companies:
            relevant_docs.extend(retriever.invoke(f"{comp} total revenue net income financial results"))
    else:
        # Standard retrieval with clean context
        search_query = query
        if chat_history and len(query.strip().split()) <= 3:
            last_user_prompt = next((m["content"] for m in reversed(chat_history) if m["role"] == "user"), "")
            search_query = f"{last_user_prompt} {query}"
        relevant_docs = retriever.invoke(search_query)

    # Deduplicate documents by content and source
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

    history_context = ""
    if chat_history:
        recent_turns = [f"{'User' if m['role'] == 'user' else 'Auditor'}: {m['content']}" for m in chat_history[-4:]]
        history_context = "\n".join(recent_turns)

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

Recent Conversation History:
{history_context}

Retrieved SEC Filings Context:
{context_text}

Latest User Query: {query}

Audit Answer:"""

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

    # Extract matching sources from the cited docs
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