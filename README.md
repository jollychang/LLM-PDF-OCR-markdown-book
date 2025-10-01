# LLM PDF OCR markdown book

## Overview
`ocr_md_book.py` turns a folder of scanned page images into clean Markdown, merges every page into a single `book.md`, and finally packages the result as an EPUB (plus optional AZW3/MOBI if Calibre is available). The OCR step relies on Alibaba DashScope (Tongyi) multimodal models and includes light post-processing to remove headers, footers, page numbers, and unwanted hard wraps. The tool is resumable and designed for macOS but remains cross-platform friendly.

## Prerequisites
1. **Python** 3.10 or newer.
2. **Python dependencies** (install inside a virtual environment if possible):
   ```bash
   python3 -m pip install httpx pillow tqdm pyyaml
   ```
3. **External tools**:
   - `pandoc` (required) – e.g. `brew install pandoc` on macOS.
   - `ebook-convert` from Calibre (optional) if you want AZW3/MOBI output.
4. **DashScope API Key** – export before running:
   ```bash
   export DASHSCOPE_API_KEY="sk-your-key"
   ```

## Converting PDF to Images (optional)
If your source is a PDF, convert it to page images first. Install Poppler (provides `pdftoppm`), e.g. `brew install poppler`, then run:
```bash
pdftoppm -png -r 300 "input.pdf" "output-prefix"
```
This will create files such as `output-prefix-01.png`, `output-prefix-02.png`, … that you can place in the images directory for the OCR step.

## Preparing Assets
- Place all page images (PNG/JPG) in one directory; natural sorting is handled automatically, but numeric suffixes are recommended.
- Provide a cover image if you want EPUB metadata to include it; the path must exist or pandoc will fail.

## Quick Start
From the project root, run:
```bash
python3 ocr_md_book.py \
  --images-dir ./book_images \
  --title "The Wealth Handbook" \
  --author "Unknown" \
  --lang zh-CN \
  --max-width 1800 \
  --concurrency 4 \
  --model qwen3-omni-flash \
  --cover ./book_images/output-001.png \
  --out-name book \
  --skip-ocr-existing \
  --to-azw3 \
  --to-mobi
```
Results land in `book_images/_out/`:
- `pages/page-0001.md`, … individual Markdown files
- `book.md` – merged document
- `book.epub` – main deliverable (and `book.azw3`/`book.mobi` when Calibre is detected and flags set)

## Key Flags
- `--images-dir` *(required)*: folder containing images.
- `--title`, `--author`, `--lang`: EPUB metadata.
- `--max-width`: downscale width before upload (never upscale).
- `--concurrency`: async OCR concurrency; start between 1–4.
- `--model`: DashScope model name (e.g. `qwen3-omni-flash`).
- `--cover`: cover image path for EPUB metadata (must exist).
- `--out-name`: output file prefix (default `book`).
- `--skip-ocr-existing`: skip pages with existing Markdown (resume support).
- `--from-list`: newline-separated file list to control ordering.
- `--pages`: subset pages like `1-50,120,121-130`.
- `--dry-run`: list pages to process without running OCR.
- `--to-azw3`, `--to-mobi`: build Kindle formats if `ebook-convert` is available.
- `--verbose`: show detailed logs (default output is concise).

## Processing Workflow
1. Gather images (or read from `--from-list`) and sort naturally.
2. Auto-rotate with EXIF data, optionally downscale, and forward to DashScope using several payload variants for compatibility.
3. Clean the resulting Markdown and write to `_out/pages/page-XXXX.md`.
4. Merge pages into `_out/book.md` with `# 第 N 页` separators.
5. Build the EPUB via pandoc, and optionally call Calibre to produce AZW3/MOBI.

## Resuming Runs
- Combine `--skip-ocr-existing` with the default output structure to resume after interruptions.
- Failed pages are logged by index; re-run the command (optionally with `--pages`) to fill the gaps.

## Troubleshooting
- **HTTP 400 “url error”**: ensure the chosen model supports base64 payloads. If it requires public URLs, upload images to accessible HTTPS locations and reference them via `--from-list`.
- **Cover file missing**: confirm the path passed to `--cover` exists or omit the flag.
- **Calibre not found**: the script logs a warning and skips AZW3/MOBI when `ebook-convert` is absent.

## License
No specific license is provided. Use internally or personally as needed, and comply with the licenses of DashScope, Calibre, pandoc, and other dependencies.
