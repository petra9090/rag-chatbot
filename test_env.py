from dotenv import load_dotenv
import os

load_dotenv()

llama_key = os.getenv("LLAMA_CLOUD_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")

print("LlamaCloud key:", llama_key[:5] + "..." if llama_key else "NOT FOUND")
print("Anthropic key:",  anthropic_key[:5] + "..." if anthropic_key else "NOT FOUND")