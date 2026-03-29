from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from github_loader import load_documents_from_github
from dotenv import load_dotenv
import os

load_dotenv()

# Settings
Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
Settings.llm = Anthropic(model="claude-sonnet-4-6")

# Load or build index
if os.path.exists("storage"):
    print("Loading existing index...")
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context)
else:
    print("Building new index from GitHub...")
    documents = load_documents_from_github()
    index = VectorStoreIndex.from_documents(
        documents,
        chunk_size=512,        # smaller chunks = more precise retrieval
        chunk_overlap=50,      # overlap prevents cutting context mid-sentence
    )
    index.storage_context.persist("storage")

# Memory buffer — remembers last 10 exchanges
memory = ChatMemoryBuffer.from_defaults(token_limit=4000)

# Chat engine with memory + better context handling
chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    system_prompt=(
        "You are a helpful study assistant. Answer questions based on the "
        "provided course notes. Be concise but thorough. If the answer is "
        "not in the notes, say so clearly."
    ),
    similarity_top_k=5,        # retrieve top 5 most relevant chunks
    verbose=False,
)

print("Chatbot ready! Type 'exit' to quit.\n")

while True:
    question = input("You: ")
    if question.lower() in ["exit", "quit"]:
        break
    response = chat_engine.chat(question)
    print(f"\nAssistant: {response}\n")