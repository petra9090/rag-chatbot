"""
github_loader.py
----------------
Fetches all .qmd / .md / .txt files from a GitHub repository and returns them
as LlamaIndex Document objects. Files are downloaded in parallel for speed.

Structure assumed:
    hs25/
      cip/
        cip_notes.qmd
    fs26/
      ...

Environment variables (add to .env):
    GITHUB_TOKEN=your_personal_access_token
    GITHUB_OWNER=nilsrechberger
    GITHUB_REPO=mscids-notes
    GITHUB_BRANCH=main
    GITHUB_FOLDERS=hs25,fs26
"""

import os
import base64
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from llama_index.core import Document
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_TOKEN   = os.getenv("GITHUB_TOKEN", "")
GITHUB_OWNER   = os.getenv("GITHUB_OWNER", "nilsrechberger")
GITHUB_REPO    = os.getenv("GITHUB_REPO",  "mscids-notes")
GITHUB_BRANCH  = os.getenv("GITHUB_BRANCH", "main")

_folders_env   = os.getenv("GITHUB_FOLDERS", "hs25,fs26")
GITHUB_FOLDERS = [f.strip() for f in _folders_env.split(",") if f.strip()]

# File types to load
EXTENSIONS = [".qmd", ".md", ".txt"]

BASE_URL = "https://api.github.com"


def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def _parse_metadata(path: str) -> dict:
    """
    Extract semester, subject and filename from a nested path.
    e.g. hs25/cip/cip_notes.qmd -> semester=hs25, subject=cip, file_name=cip_notes.qmd
    """
    parts = path.split("/")
    return {
        "semester":  parts[0] if len(parts) > 0 else "",
        "subject":   parts[1] if len(parts) > 1 else "",
        "file_name": parts[-1],
        "file_path": path,
    }


def _list_files(path: str) -> list[dict]:
    """Recursively list every file under path."""
    url = f"{BASE_URL}/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{path}"
    response = requests.get(url, headers=_headers(), params={"ref": GITHUB_BRANCH})

    if response.status_code == 404:
        print(f"   WARNING: '{path}' not found in repo — check GITHUB_FOLDERS in .env")
        return []

    response.raise_for_status()

    files = []
    for item in response.json():
        if item["type"] == "file":
            files.append(item)
        elif item["type"] == "dir":
            files.extend(_list_files(item["path"]))
    return files


def _fetch_content(file_info: dict) -> str:
    """Download and base64-decode a file from GitHub."""
    response = requests.get(file_info["url"], headers=_headers())
    response.raise_for_status()
    return base64.b64decode(response.json()["content"]).decode("utf-8")


def _fetch_one(file_info: dict) -> Document:
    """Fetch a single file and return as a LlamaIndex Document."""
    content  = _fetch_content(file_info)
    metadata = _parse_metadata(file_info["path"])
    metadata["source_url"] = file_info["html_url"]
    metadata["branch"]     = GITHUB_BRANCH
    return Document(text=content, metadata=metadata)


def load_documents_from_github(verbose: bool = True) -> list[Document]:
    """
    Crawl GITHUB_FOLDERS, fetch all .qmd / .md / .txt files in parallel,
    and return them as LlamaIndex Documents with semester + subject metadata.
    """
    if verbose:
        print(f"\nCrawling github.com/{GITHUB_OWNER}/{GITHUB_REPO}  "
              f"[branch: {GITHUB_BRANCH}]")
        print(f"Looking in folders : {GITHUB_FOLDERS}")
        print(f"File types         : {EXTENSIONS}\n")

    # ── 1. Collect all file listings ─────────────────────────────────────────
    all_files = []
    for folder in GITHUB_FOLDERS:
        print(f"  Scanning {folder}/...")
        found = _list_files(folder)
        print(f"    {len(found)} total files in {folder}/")
        all_files.extend(found)

    # ── 2. Filter by extension ────────────────────────────────────────────────
    matched = [f for f in all_files
               if any(f["name"].lower().endswith(ext) for ext in EXTENSIONS)]

    print(f"\n  Matched {len(matched)} note file(s) "
          f"({', '.join(EXTENSIONS)}) out of {len(all_files)} total\n")

    if len(matched) == 0:
        print("  No files found — double-check:")
        print(f"    GITHUB_OWNER   = {GITHUB_OWNER}")
        print(f"    GITHUB_REPO    = {GITHUB_REPO}")
        print(f"    GITHUB_FOLDERS = {GITHUB_FOLDERS}")
        return []

    # ── 3. Download files in parallel (faster than sequential) ───────────────
    documents = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_fetch_one, f): f for f in matched}
        for future in as_completed(futures):
            try:
                doc = future.result()
                documents.append(doc)
                if verbose:
                    print(f"  OK  [{doc.metadata['semester']}]  "
                          f"{doc.metadata['subject']} / {doc.metadata['file_name']}")
            except Exception as e:
                print(f"  SKIP {futures[future]['path']} — {e}")

    # ── 4. Summary ────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*55}")
        print(f"Loaded {len(documents)} document(s) from GitHub")
        for folder in GITHUB_FOLDERS:
            count = sum(1 for d in documents
                        if d.metadata.get("semester") == folder)
            print(f"  {folder}: {count} file(s)")
        print(f"{'='*55}\n")

    return documents