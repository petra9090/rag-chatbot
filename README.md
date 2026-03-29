# rag-chatbot
Overview
This chatbot lets students ask questions about their course notes using Retrieval-Augmented Generation (RAG). Documents are indexed locally, and Claude answers questions based only on the uploaded material.

Project Structure
rag-chatbot/
├── app.py             # Streamlit web interface
├── main.py            # Command-line interface
├── evaluate.py        # RAGAS evaluation script
├── test_env.py        # API key verification helper
├── requirements.txt    # Python dependencies
├── .env               # API keys (not committed)
├── data/              # Place your .md course notes here
└── storage/           # Auto-generated index cache

Prerequisites
Python 3.13 or higher
An Anthropic API key  (https://console.anthropic.com)
A LlamaCloud API key  (https://cloud.llamaindex.ai)

Installation
1. Clone the repository
git clone https://github.com/petra9090/rag-chatbot.git
cd rag-chatbot

2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

3. Install dependencies
pip install -r requirements.txt

4. Configure environment variables
Create a .env file in the project root and add your API keys:
ANTHROPIC_API_KEY=your_anthropic_key_here
LLAMA_CLOUD_API_KEY=your_llamacloud_key_here
Tip: Run python test_env.py to verify your keys are loaded correctly.

Preparing the course notes
Place the course notes in the data/ folder. The chatbot reads Markdown (.md) files.

Converting .qmd files to .md
If your notes are in Quarto (.qmd) format, rename them to .md before running the app.

macOS / Linux
for f in ./data/*.qmd; do mv "$f" "${f%.qmd}.md"; done

Windows (PowerShell)
Get-ChildItem -Path .\data -Filter *.qmd | Rename-Item -NewName { $_.Name -replace '\.qmd$', '.md' }

After conversion, all .md files in data/ will be indexed automatically on first launch.

Running the Application
Streamlit web interface  (recommended)
streamlit run app.py
Open http://localhost:8501 in your browser. The index is built on first run and cached in the storage/ folder.

Command-line interface
python main.py
Type your question and press Enter. Type exit or quit to stop.

Evaluating Answer Quality
Run the RAGAS evaluation suite to measure how well the chatbot answers a set of reference questions:
python evaluate.py
Results are printed to the terminal and saved to ragas_results.json. Metrics reported:
Faithfulness — are claims in the answer supported by the retrieved context?
Answer Relevancy — does the answer address the question?
Context Precision — are the retrieved chunks actually useful?
Context Recall — does the retrieved context cover the ground-truth answer?