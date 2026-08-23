import os
import time
import requests
import streamlit as st
from app.retriever import get_retriever

SYSTEM_PROMPT = """You are AuditMind AI, an enterprise-grade financial auditing assistant specialized in SEC 10-K filings.
Analyze the provided context documents and immediate prior conversation turn carefully to answer the user's inquiry accurately.

STRICT AUDIT RULES:
1. Grounding: Rely ONLY on the provided Context and Immediate Prior Turn. Do NOT use outside general knowledge or hallucinate.
2. Disambiguation: If the current query asks for financial metrics without specifying a company, and no company is identifiable from the immediately preceding message, you MUST respond: "Which company's filings would you like me to audit? (e.g., Apple, Microsoft, NVIDIA)".
3. Direct Output: Output ONLY the final audit response. Do not output your thinking, scratchpads, or analysis of rules.
4. Precision: Cite exact fiscal years and monetary figures directly from the source text. Look closely at tables titled Consolidated Statements of Income / Operations / Net Sales.
5. Source Attribution: Always specify the source document name and page number for every data point."""

def generate_answer(query: str, chat_history: list = None):
    retriever = get_retriever(k=10)
    
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
            relevant_docs.extend(retriever.invoke(f"{comp} Consolidated Statements of Income Operations total revenue net income"))
    else:
        search_query = query
        if last_user_query and len(query.strip().split()) <= 3:
            search_query = f"{query} {last_user_query} Consolidated Statements of Income Operations total revenue net sales"
        relevant_docs = retriever.invoke(search_query)

    # Deduplicate documents
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

    user_prompt = f"""{SYSTEM_PROMPT}

[IMMEDIATE PREVIOUS TURN]
{history_context if history_context else 'None'}

[CONTEXT DOCUMENTS]
{context_text}

[USER QUESTION]
{query}

Audit Answer:"""

    payload = {
        "contents": [
            {"parts": [{"text": user_prompt}]}
        ],
        "generationConfig": {
            "temperature": 0.1
        }
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={api_key}"
    
    answer_text = None
    
    # Auto-retry loop with exponential backoff on 429 rate limits
    for attempt in range(3):
        try:
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
            
            if res.status_code == 200:
                data = res.json()
                candidates = data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    clean_parts = [p["text"] for p in parts if not p.get("thought", False) and p.get("text")]
                    
                    if clean_parts:
                        answer_text = "\n".join(clean_parts).strip()
                    elif parts:
                        answer_text = parts[-1].get("text", "").strip()
                break
            elif res.status_code == 429:
                # Sleep and retry automatically
                if attempt < 2:
                    time.sleep(6 * (attempt + 1))
                    continue
                else:
                    answer_text = "API rate limit reached. Please wait 15 seconds before submitting your next prompt."
            else:
                answer_text = f"API Error ({res.status_code}): {res.text}"
                break
        except requests.exceptions.Timeout:
            answer_text = "API request timed out. Please retry in a moment."
            break
        except Exception as e:
            answer_text = f"Error processing audit query: {str(e)}"
            break

    if answer_text and "Rule 2" in answer_text and "Disambiguation" in answer_text:
        answer_text = "Which company's filings would you like me to audit? (e.g., Apple, Microsoft, NVIDIA)"

    used_sources = []
    for doc in unique_docs:
        source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
        page_num = doc.metadata.get("page", 0) + 1
        if source_name in (answer_text or ""):
            used_sources.append(f"Document Name: {source_name} | Page Number: {page_num}")

    final_sources = list(set(used_sources)) if used_sources else list(set([
        f"Document Name: {os.path.basename(d.metadata.get('source', 'Unknown'))} | Page Number: {d.metadata.get('page', 0) + 1}"
        for d in unique_docs
    ]))

    return {
        "answer": answer_text,
        "sources": final_sources
    }