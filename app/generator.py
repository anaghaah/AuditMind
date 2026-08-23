import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from app.retriever import get_retriever

SYSTEM_PROMPT = """
You are AuditMind AI, an enterprise-grade financial auditing assistant specialized in SEC 10-K filings.
Analyze the provided context documents carefully and answer the user's inquiry accurately.

STRICT AUDIT RULES:
1. Grounding: Rely ONLY on the provided Context. Do NOT use outside general knowledge or hallucinate.
2. Company Ambiguity: If the user asks a metric question (e.g., revenue, net income, cash flow) WITHOUT specifying a company name, and the context contains multiple entities or is unclear, do NOT assume. Explicitly ask the user: "Which company's filings would you like me to audit?"
3. Precision: Cite exact fiscal years and monetary figures directly from the source text.
4. Source Attribution: Always specify the source document name and page number for every data point.

Context:
{context}

Question:
{question}

Audit Answer:
"""

def generate_answer(query: str):
    retriever = get_retriever()
    relevant_docs = retriever.invoke(query)

    context_text = ""
    for doc in relevant_docs:
        source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
        page_num = doc.metadata.get("page", 0) + 1
        context_text += f"\n[Document: {source_name} | Page: {page_num}]\n{doc.page_content}\n"

    # Fetch API Key securely
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.0,
        google_api_key=api_key
    )

    formatted_prompt = SYSTEM_PROMPT.format(context=context_text, question=query)
    response = llm.invoke(formatted_prompt)

    sources = [
        f"Document Name: {os.path.basename(doc.metadata.get('source', 'Unknown'))} | Page Number: {doc.metadata.get('page', 0) + 1}"
        for doc in relevant_docs
    ]

    return {
        "answer": response.content,
        "sources": list(set(sources))
    }