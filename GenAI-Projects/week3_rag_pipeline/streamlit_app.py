"""
Streamlit Web Interface for RAG Legal Assistant
------------------------------------------------
Frontend for the Flask-based RAG Legal Assistant.
Allows users to:
 - Ask legal questions
 - View AI-generated answers with citations
 - See retrieved sources
 - Review evaluation results

Requires Flask backend (api.py) running locally or remotely.
"""

import streamlit as st
import requests
import json

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="RAG Legal Assistant",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ RAG Legal Assistant")
st.caption("Ask legal questions and receive context-grounded answers with citations.")

# -------------------------------
# Sidebar Configuration
# -------------------------------
st.sidebar.header("Settings")

api_url = st.sidebar.text_input(
    "Backend API URL",
    value="http://localhost:8000/query",
    help="Enter the Flask API endpoint (default: local)."
)

st.sidebar.divider()
st.sidebar.write("🧠 Powered by LangChain + Hybrid RAG Retrieval")

# -------------------------------
# User Input
# -------------------------------
st.markdown("### 🔍 Ask a Legal Question")

question = st.text_area(
    "Enter your legal question below:",
    placeholder="e.g., What is the punishment for cheating under the Indian Penal Code?",
    height=120
)

submit = st.button("Submit", use_container_width=True)
clear = st.button("Clear", use_container_width=True)

if clear:
    st.session_state.clear()

# -------------------------------
# API Call
# -------------------------------
if submit and question.strip():
    with st.spinner("Retrieving and generating answer..."):
        try:
            response = requests.post(api_url, json={"question": question}, timeout=60)
            if response.status_code == 200:
                data = response.json()

                # Display main answer
                st.markdown("### 🧾 Answer")
                st.success(data.get("answer", "No answer generated."))

                # Display sources
                sources = data.get("sources", [])
                if sources:
                    st.markdown("### 📚 Sources")
                    for i, src in enumerate(sources, 1):
                        with st.expander(f"Source [{i}]"):
                            st.markdown(f"**Content:** {src.get('page_content', '')}")
                            metadata = src.get("metadata", {})
                            if metadata:
                                st.json(metadata)

                # Display evaluation
                evaluation = data.get("evaluation", {})
                if evaluation:
                    st.markdown("### ✅ Evaluation Summary")
                    st.json(evaluation)

            else:
                st.error(f"Error: Received status code {response.status_code}")
                st.text(response.text)

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to API: {e}")
else:
    if not question.strip() and submit:
        st.warning("Please enter a question before submitting.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Developed with ❤️ using Flask + LangChain + Streamlit | © 2025")