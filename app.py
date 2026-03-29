import streamlit as st
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from github_loader import load_documents_from_github
from dotenv import load_dotenv
import os
import shutil

load_dotenv()


# Page config
st.set_page_config(page_title="Study Assistant", page_icon="📚", layout="centered")
st.title("📚 Study Assistant")
st.caption("Ask me anything about your course notes!")

# Sidebar: refresh button
with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("🔄 Refresh notes from GitHub"):
        if os.path.exists("storage"):
            shutil.rmtree("storage")
        st.cache_resource.clear()
        st.success("Cache cleared — reloading latest notes...")
        st.rerun()


# Load models and index once (cached so it doesn't reload on every message)
@st.cache_resource
def load_chat_engine():
    Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    Settings.llm = Anthropic(model="claude-sonnet-4-6")

    if os.path.exists("storage"):
        storage_context = StorageContext.from_defaults(persist_dir="storage")
        index = load_index_from_storage(storage_context)
    else:
        documents = load_documents_from_github()
        if not documents:
            st.error("No documents found. Check your GITHUB_* settings in .env")
            st.stop()
        index = VectorStoreIndex.from_documents(documents, chunk_size=512, chunk_overlap=50)
        index.storage_context.persist("storage")

    memory = ChatMemoryBuffer.from_defaults(token_limit=4000)

    return index.as_chat_engine(
        chat_mode="condense_plus_context",
        memory=memory,
        system_prompt=(
            "You are a helpful study assistant. Answer questions based on the "
            "provided course notes. Be concise but thorough. If the answer is "
            "not in the notes, say so clearly."
        ),
        similarity_top_k=5,
    )


chat_engine = load_chat_engine()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your notes..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.markdown(str(response))

    st.session_state.messages.append({"role": "assistant", "content": str(response)})