# rag-chatbot

This chatbot lets students ask questions about their course notes using Retrieval-Augmented Generation (RAG). Documents are indexed locally, and Claude answers questions based only on the uploaded material.

---

## Project Structure

```
rag-chatbot/
├── app.py              # Streamlit web interface
├── main.py             # Command-line interface
├── evaluate.py         # RAGAS evaluation script
├── test_env.py         # API key verification helper
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed)
├── data/               # Place your .md course notes here
└── storage/            # Auto-generated index cache
```

---

## Prerequisites

- Python 3.13 or higher
- An Anthropic API key — [console.anthropic.com](https://console.anthropic.com)
- A LlamaCloud API key — [cloud.llamaindex.ai](https://cloud.llamaindex.ai)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/petra9090/rag-chatbot.git
cd rag-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_anthropic_key_here
LLAMA_CLOUD_API_KEY=your_llamacloud_key_here
```

> **Tip:** Run `python test_env.py` to verify your keys are loaded correctly.

---

## 📄 Preparing the course notes

Place your course notes as `.md` files in the `data/` folder.

### Converting `.qmd` files to `.md`

**macOS / Linux:**
```bash
for f in ./data/*.qmd; do mv "$f" "${f%.qmd}.md"; done
```

**Windows (PowerShell):**
```powershell
Get-ChildItem -Path .\data -Filter *.qmd | Rename-Item -NewName { $_.Name -replace '\.qmd$', '.md' }
```

---

## Running the application

### Streamlit web interface (recommended)

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.  
The index is built on first run and cached in the `storage/` folder.

### Command-line interface

```bash
python main.py
```

Type your question and press Enter. Type `exit` or `quit` to stop.

---

## Evaluating answer quality

```bash
python evaluate.py
```

Results are saved to `ragas_results.json`. Metrics reported:

| Metric | Description |
|---|---|
| Faithfulness | Are claims in the answer supported by the retrieved context? |
| Answer Relevancy | Does the answer address the question? |
| Context Precision | Are the retrieved chunks actually useful? |
| Context Recall | Does the retrieved context cover the ground-truth answer? |

---
