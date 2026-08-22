import sys
import os

# project root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from app.generator import generate_answer

# Page Configuration
st.set_page_config(
    page_title="AuditMind AI",
    page_icon="📊",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚡ AuditMind AI")
    st.caption("10-K Compliance & Financial Audit Engine")
    st.markdown("---")
    st.write("📌 **System Capabilities:**")
    st.markdown("""
    - Financial Extraction (Revenue, Net Income)
    - Risk & Obligation Audits
    - Zero Hallucination Retrieval
    """)
    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Header
st.title("📊 Financial Document Auditor")
st.write("Ask any questions regarding the processed 10-K financial reports.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your AuditMind AI auditor. How can I assist you with your document analysis today?"}
    ]

# Display Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input Box
if prompt := st.chat_input("Ask a financial question (e.g., What is the total revenue for 2025?)..."):
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Auditing document embeddings..."):
            response = generate_answer(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})