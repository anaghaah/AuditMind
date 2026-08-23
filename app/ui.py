import os
import sys
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from app.generator import generate_answer
except ImportError:
    from generator import generate_answer

st.set_page_config(page_title="AuditMind AI", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stChatInputContainer { padding-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⚡ AuditMind AI")
    st.caption("10-K Compliance & Financial Audit Engine")
    st.markdown("---")
    st.markdown("📌 **System Capabilities:**")
    st.markdown("- Financial Extraction (Revenue, Net Income)")
    st.markdown("- Risk & Obligation Audits")
    st.markdown("- Zero Hallucination Retrieval")
    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("📊 Financial Document Auditor")
st.write("Ask any questions regarding the processed 10-K financial reports.")

# Initialize persistent message session state
if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your AuditMind AI auditor. How can I assist you with your document analysis today?",
            "sources": []
        }
    ]

# Display all prior messages in state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown("**Sources & Citations:**")
            for src in msg["sources"]:
                st.markdown(f"- {src}")

# Handle new user input
if prompt := st.chat_input("Ask a financial question (e.g., What is the total revenue for 2025?)..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing SEC 10-K filings..."):
            result = generate_answer(prompt, chat_history=st.session_state.messages)
            answer = result["answer"]
            sources = result["sources"]

            st.markdown(answer)
            if sources:
                st.markdown("**Sources & Citations:**")
                for src in sources:
                    st.markdown(f"- {src}")

    # Append interaction to persistent history
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })