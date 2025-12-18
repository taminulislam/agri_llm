#!/usr/bin/env python3
"""
Chandra OCR: PDF to Text Extraction
Converts agricultural textbook PDFs to JSONL format for LLM fine-tuning.
"""

import json
import pathlib
import subprocess
from datetime import datetime

# ==================== CONFIGURATION ====================
PDF_DIR = pathlib.Path("")
OUT_DIR = pathlib.Path("")

# Choose method: "vllm" (default) or "hf" (Hugging Face)
METHOD = "hf"  # switch to "vllm" if you have a vLLM server running
PAGE_RANGE = None  # e.g., "1-5,7" to limit pages
BATCH_SIZE = 1     # pages per batch; increase to speed up if memory allows

# Chunking settings
WORDS_PER_CHUNK = 800 
OVERLAP = 120
# =======================================================


def run_chandra(pdf_path: pathlib.Path):
    """Run Chandra OCR on a single PDF file."""
    cmd = [
        "chandra",
        str(pdf_path),
        str(OUT_DIR),
        "--method", METHOD,
        "--no-images",
    ]
    if PAGE_RANGE:
        cmd += ["--page-range", PAGE_RANGE]
    if BATCH_SIZE:
        cmd += ["--batch-size", str(BATCH_SIZE)]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def collect_to_jsonl():
    """Collect Chandra outputs into page-level JSONL."""
    jsonl_path = OUT_DIR / "dataset_pages.jsonl"
    
    with jsonl_path.open("w", encoding="utf-8") as f_out:
        # Use recursive glob to find metadata files in subdirectories
        for meta_file in OUT_DIR.glob("**/*_metadata.json"):
            with meta_file.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            
            doc_base = meta_file.name.replace("_metadata.json", "")
            # Look for .md file in the same directory as the metadata file
            md_file = meta_file.parent / f"{doc_base}.md"
            text = md_file.read_text(encoding="utf-8")
            
            # Split by page markers if present
            pages = text.split("\n---\n") if "\n---\n" in text else [text]
            
            for page_idx, page_text in enumerate(pages, start=1):
                record = {
                    "id": f"{doc_base}-p{page_idx}",
                    "doc": doc_base,
                    "page": page_idx,
                    "text": page_text.strip(),
                    "source": str((OUT_DIR / meta_file.name).resolve()),
                    "ingested_at": datetime.utcnow().isoformat() + "Z",
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Wrote JSONL: {jsonl_path}")
    return jsonl_path


def chunk_text(text):
    """Split text into overlapping chunks."""
    words = text.split()
    step = max(1, WORDS_PER_CHUNK - OVERLAP)
    for i in range(0, len(words), step):
        yield " ".join(words[i:i + WORDS_PER_CHUNK])


def create_chunks(jsonl_path):
    """Create chunked dataset from page-level JSONL."""
    chunk_jsonl = OUT_DIR / "dataset_chunks.jsonl"
    
    with open(jsonl_path, "r", encoding="utf-8") as fin, \
         open(chunk_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            for idx, chunk in enumerate(chunk_text(rec["text"]), start=1):
                out = {
                    "id": f'{rec["id"]}-c{idx}',
                    "doc": rec["doc"],
                    "page": rec["page"],
                    "chunk_id": idx,
                    "text": chunk,
                    "source": rec["source"],
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
    
    print(f"Wrote chunks: {chunk_jsonl}")


def main():
    """Main entry point."""
    # Create output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all PDFs
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    assert pdfs, f"No PDFs found in {PDF_DIR}"
    print(f"Found {len(pdfs)} PDF files")
    
    # Step 1: Run Chandra OCR on all PDFs
    print("\n=== Step 1: Running Chandra OCR ===")
    for pdf in pdfs:
        print(f"\nProcessing: {pdf.name}")
        run_chandra(pdf)
    
    # Step 2: Collect outputs into JSONL
    print("\n=== Step 2: Creating page-level JSONL ===")
    jsonl_path = collect_to_jsonl()
    
    # Step 3: Create chunked dataset
    print("\n=== Step 3: Creating chunked JSONL ===")
    create_chunks(jsonl_path)
    
    print("\n=== Done! ===")
    print(f"Outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
