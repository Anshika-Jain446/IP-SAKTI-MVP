"""
IP-SAKTI corpus ingestion script.

Rebuilds backend/data/corpus.json from ALL PDFs found under:

    backend/data/ip/    -> domain "IP"
    backend/data/tk/    -> domain "TK"   (searched recursively --
                                            handles the nested
                                            ip_sakti_tk_pipeline/... path)
    backend/data/abs/   -> domain "ABS"

Run this from inside the `backend/` folder (same folder as main.py):

    pip install pypdf
    python ingest_corpus.py

Output: backend/data/corpus.json (overwritten)

This replaces whatever manual/partial process built corpus.json before.
It is safe to re-run any time you add new PDFs.
"""

import json
import os
import re
import sys
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    print("Missing dependency. Run: pip install pypdf")
    sys.exit(1)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "corpus.json"

# folder name (under data/) -> domain tag used everywhere in main.py
DOMAIN_FOLDERS = {
    "ip": "IP",
    "tk": "TK",
    "abs": "ABS",
}

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def clean_text(text):
    """Collapse whitespace/newlines the same way the existing corpus looks."""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_text(pdf_path):
    """Extract and concatenate text from every page of a PDF."""
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        print(f"  [SKIP] Could not open {pdf_path.name}: {e}")
        return ""

    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception as e:
            print(f"  [WARN] Could not extract a page in {pdf_path.name}: {e}")

    return clean_text(" ".join(pages_text))


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple sliding-window chunking, matching the existing corpus style."""
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)
    step = max(1, chunk_size - overlap)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start += step

    return chunks


def find_pdfs_recursive(folder):
    """Recursively find every .pdf under a folder (handles nested paths)."""
    if not folder.exists():
        print(f"  [MISSING] Folder does not exist: {folder}")
        return []
    return sorted(folder.rglob("*.pdf"))


def build_corpus():
    corpus = []
    summary = {}

    for folder_name, domain in DOMAIN_FOLDERS.items():
        folder = DATA_DIR / folder_name
        pdf_files = find_pdfs_recursive(folder)

        print(f"\n=== Domain: {domain}  (folder: {folder}) ===")
        print(f"Found {len(pdf_files)} PDF file(s).")

        domain_chunk_count = 0

        for pdf_path in pdf_files:
            filename = pdf_path.name
            print(f"  Processing: {filename}")

            text = extract_pdf_text(pdf_path)
            if not text:
                print(f"  [SKIP] No extractable text in {filename}")
                continue

            chunks = chunk_text(text)
            print(f"    -> {len(chunks)} chunk(s)")

            stem = filename.replace(".pdf", "").replace(".PDF", "")

            for i, chunk in enumerate(chunks):
                corpus.append({
                    "id": f"{domain.lower()}-{stem}-{i}",
                    "domain": domain,
                    "source": filename,
                    "page_content": chunk,
                    "metadata": {
                        "filename": filename,
                        "domain": domain,
                        "chunk": i,
                    },
                })
                domain_chunk_count += 1

        summary[domain] = {
            "files": len(pdf_files),
            "chunks": domain_chunk_count,
        }

    return corpus, summary


def main():
    print(f"Base dir:   {BASE_DIR}")
    print(f"Data dir:   {DATA_DIR}")
    print(f"Output:     {OUTPUT_PATH}")

    if not DATA_DIR.exists():
        print(f"\n[ERROR] {DATA_DIR} does not exist. "
              f"Run this script from inside your backend/ folder.")
        sys.exit(1)

    corpus, summary = build_corpus()

    if not corpus:
        print("\n[ERROR] No chunks were produced. corpus.json was NOT written. "
              "Check the folder paths above.")
        sys.exit(1)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)
    for domain, stats in summary.items():
        print(f"{domain}: {stats['files']} file(s), {stats['chunks']} chunk(s)")
    print(f"Total chunks written: {len(corpus)}")
    print(f"Written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()