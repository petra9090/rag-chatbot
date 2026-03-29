# rag-chatbot

An educational study assistant powered by LlamaIndex + Claude.  
This chatbot lets students ask questions about their course notes using Retrieval-Augmented Generation (RAG). Notes are crawled directly from a GitHub repository, so the chatbot always uses the latest version.

---

## Project Structure

```
rag-chatbot/
├── app.py              # Streamlit web interface
├── main.py             # Command-line interface
├── github_loader.py    # Crawls notes from GitHub (replaces static data/ folder)
├── evaluate.py         # RAGAS evaluation script
├── test_env.py         # API key verification helper
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed — see setup below)
└── storage/            # Auto-generated index cache (not committed)
```

---

## Prerequisites

- Python 3.13 or higher
- An Anthropic API key — [console.anthropic.com](https://console.anthropic.com)
- A LlamaCloud API key — [cloud.llamaindex.ai](https://cloud.llamaindex.ai)
- A GitHub Personal Access Token — [github.com/settings/tokens](https://github.com/settings/tokens)

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

Create a `.env` file in the project root — **never commit this file**:

```
ANTHROPIC_API_KEY=your_anthropic_key_here
LLAMA_CLOUD_API_KEY=your_llamacloud_key_here

GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
GITHUB_OWNER=nilsrechberger
GITHUB_REPO=mscids-notes
GITHUB_BRANCH=main
GITHUB_FOLDERS=hs25,fs26
```

> **Tip:** Run `python test_env.py` to verify your keys are loaded correctly.

---

## How Notes are loaded

Notes are fetched directly from GitHub at runtime — there is no local `data/` folder.  
The loader (`github_loader.py`) recursively crawls the folders defined in `GITHUB_FOLDERS` and reads all `.qmd`, `.md`, and `.txt` files.

**Supported folder structure:**

```
hs25/
  cip/
    cip_notes.qmd
  statistics/
    stats_notes.qmd
fs26/
  databases/
    db_notes.qmd
```

Each document is indexed with metadata: **semester**, **subject**, and **filename**.

---

## Running the application

### Streamlit web interface (recommended)

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

On first launch the app will:
1. Crawl GitHub and fetch all notes
2. Build the vector index
3. Cache it in `storage/` for fast subsequent starts

### Refreshing notes

When new notes are pushed to the GitHub repo, click **🔄 Refresh notes from GitHub** in the sidebar to re-crawl and rebuild the index.

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
