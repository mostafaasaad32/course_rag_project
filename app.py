import streamlit as st
import os
import sys

# Ensure the src directory is in the path so relative imports inside src work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.retriever_llm import ask_course_assistant
from src.embeddings import create_vector_db

st.set_page_config(page_title="Course Assistant", page_icon="📚", layout="wide")

st.title("📚 Course Assistant (RAG)")
st.write("Ask any question related to the course materials!")

# Sidebar for Admin operations
# with st.sidebar:
#     st.header("⚙️ Admin Settings")
#     st.write("Update the knowledge base when new documents are added to `data/raw_courses`.")
#     if st.button("Update Vector Database"):
#         with st.spinner("Processing documents and creating vector database..."):
#             try:
#                 create_vector_db()
#                 st.success("Vector database updated successfully!")
#             except Exception as e:
#                 st.error(f"Error updating database: {e}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for doc in message["sources"]:
                    source_file = os.path.basename(doc.metadata.get('source', 'Unknown'))
                    page = doc.metadata.get('page', 'N/A')
                    st.write(f"- **File:** {source_file} (Page {page})")
                    st.text(doc.page_content[:200] + "...")

# Quick questions
clicked_question = None
if not st.session_state.messages:
    st.write("### Quick Questions")
    quick_questions = [
        "What is a data pipeline?",
        "What is NLTK used for?",
        "Explain the principle of mathematical induction."
    ]
    cols = st.columns(len(quick_questions))
    for i, col in enumerate(cols):
        if col.button(quick_questions[i], use_container_width=True):
            clicked_question = quick_questions[i]

# Accept user input
prompt = st.chat_input("e.g. What is a data pipeline?")
if clicked_question:
    prompt = clicked_question

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = ask_course_assistant(prompt)
                st.markdown(answer)
                
                with st.expander("View Sources"):
                    for doc in sources:
                        source_file = os.path.basename(doc.metadata.get('source', 'Unknown'))
                        page = doc.metadata.get('page', 'N/A')
                        st.write(f"- **File:** {source_file} (Page {page})")
                        st.text(doc.page_content[:200] + "...")
                        
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
            except Exception as e:
                st.error(f"An error occurred: {e}")
