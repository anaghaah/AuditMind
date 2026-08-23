import os
import requests
import streamlit as st
from app.retriever import get_retriever

SYSTEM_PROMPT = """You are AuditMind AI, an enterprise-grade financial auditing assistant specialized in SEC 10-K filings.
Analyze the provided context documents and immediate prior conversation turn carefully to answer the user's inquiry accurately.

STRICT AUDIT RULES:
1. Grounding: Rely ONLY on the provided Context and Immediate Prior Turn. Do NOT use outside general knowledge or hallucinate.
2. Disambiguation: If the current query asks for financial metrics without specifying a company, and no company is identifiable from the immediately preceding message, you MUST respond: "Which company's filings would you like me to audit? (e.g., Apple, Microsoft, NVIDIA)".
3. Precision: Cite exact fiscal years and monetary figures directly from the source text.
4. Source Attribution: Always specify the source document name and page number for every data point."""

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

    user_prompt = f"""[IMMEDIATE PREVIOUS TURN]
{history_context if history_context else 'None'}

[CONTEXT DOCUMENTS]
{context_text}

[USER QUESTION]
{query}"""

    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "thinkingConfig": {
                "thinkingBudget": 0
            }
        }
    }

    # Discover valid active models for this API Key
    available_models = []
    try:
        models_resp = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
            timeout=10
        )
        if models_resp.status_code == 200:
            for item in models_resp.json().get("models", []):
                if "generateContent" in item.get("supportedGenerationMethods", []):
                    available_models.append(item["name"])
    except Exception:
        pass

    if not available_models:
        available_models = ["models/gemini-3.6-flash"]

    answer_text = None
    last_err_msg = ""

    for model_name in available_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={api_key}"
        res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
        
        # Fallback without thinkingConfig if model doesn't accept thinkingBudget parameter
        if res.status_code == 400 and "thinking" in res.text.lower():
            fallback_payload = {
                "system_instruction": payload["system_instruction"],
                "contents": payload["contents"],
                "generationConfig": {"temperature": 0.1}
            }
            res = requests.post(url, json=fallback_payload, headers={"Content-Type": "application/json"}, timeout=30)

        if res.status_code == 200:
            data = res.json()
            candidates = data.get("candidates", [])
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                
                # 1. Filter out parts that are marked as internal thoughts
                non_thought_parts = [p.get("text", "") for p in parts if not p.get("thought", False) and p.get("text")]
                
                if non_thought_parts:
                    answer_text = "\n".join(non_thought_parts).strip()
                elif parts:
                    # Fallback to last part if no explicit flag is set
                    answer_text = parts[-1].get("text", "").strip()

            if answer_text:
                break
        elif res.status_code == 429:
            last_err_msg = "Rate limit reached on free tier. Please wait 15–20 seconds and try again."
            continue
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