#!/usr/bin/env python3
"""\
Usage example:
    python ocr_md_book.py \
        --images-dir ./book_images \
        --title "My Scanned Book" \
        --author "Unknown"

This tool OCRs scanned page images into Markdown, merges them, and exports EPUB (plus optional AZW3/MOBI).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import httpx
from PIL import Image, ImageFile, ImageOps, PngImagePlugin

if hasattr(Image, "Resampling"):
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
else:  # pragma: no cover - older Pillow
    RESAMPLE_LANCZOS = Image.LANCZOS

Image.DEBUG = False
PngImagePlugin.DEBUG = 0
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm

OCR_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generate"
COMPATIBLE_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
OCR_PROMPT = """\
You are a typesetting expert converting page scans into readable ebooks. Transcribe the image into GitHub-Flavored Markdown (GFM) and follow these rules:
1) Do not hallucinate content; keep the natural reading order.
2) Use # / ## / ### for true headings only.
3) Preserve emphasis with **bold** and *italic*.
4) Format lists with - or numbered lists; indent two spaces per level.
5) Render tables with standard Markdown table syntax (| th | th |).
6) Keep code/formulas as ``` fenced ``` blocks or `inline`, and $...$ / $$...$$ for math.
7) Remove headers, footers, page numbers, watermarks, and obvious scan noise.
8) Leave a blank line between paragraphs; avoid forced line breaks in normal sentences.
9) Keep figure captions or footnotes as plain text or *italic* lines.
Return pure Markdown text only, with no explanations or JSON.
"""
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
HEADER_PATTERNS = [
    re.compile(r"^[Pp]age\s*\d+$"),
    re.compile(r"^\d{1,4}$"),
    re.compile(r"^(confidential|draft|company\s+name|proprietary).*$", re.IGNORECASE),
    re.compile(r"^copyright\s+reserved.*$", re.IGNORECASE),
    re.compile(r"^\u7b2c?\s*\d+\s*[\u9875\u9801]$"),
]
RATE_LIMIT_RANGE = (0.05, 0.15)
MAX_RETRIES = 5
BACKOFF_BASE = 1.0
BACKOFF_JITTER = 0.5


logger = logging.getLogger("ocr_md_book")


@dataclass
class PageTask:
    index: int
    page_number: int
    image_path: Path
    md_path: Path


@dataclass
class PageResult:
    page_number: int
    status: str
    detail: Optional[str] = None


def setup_logging(verbose: bool = False) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)

    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    root = logging.getLogger()
    root.setLevel(logging.WARNING)

    for name in ("httpx", "httpcore", "asyncio", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)


def parse_pages_spec(spec: str, total: int) -> List[int]:
    allowed: set[int] = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if start > end:
                start, end = end, start
            allowed.update(range(start, end + 1))
        else:
            allowed.add(int(token))
    filtered = sorted(x for x in allowed if 1 <= x <= total)
    return filtered


def natural_key(path: Path) -> List[Tuple[bool, str]]:
    parts = re.split(r"(\d+)", path.stem)
    key: List[Tuple[bool, str]] = []
    for part in parts:
        if part.isdigit():
            key.append((True, f"{int(part):010d}"))
        else:
            key.append((False, part))
    key.append((False, path.suffix.lower()))
    return key


def gather_images(images_dir: Path, from_list: Optional[Path]) -> List[Path]:
    if from_list:
        base_dir = from_list.parent
        items: List[Path] = []
        for raw_line in from_list.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            candidate = (base_dir / line).expanduser()
            if not candidate.is_absolute():
                candidate = (base_dir / line).resolve()
            if not candidate.exists():
                raise FileNotFoundError(f"File listed in manifest not found: {candidate}")
            if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            items.append(candidate)
        if not items:
            raise ValueError("No valid image files found in the provided list")
        return items

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
    items = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not items:
        raise ValueError("No PNG/JPG images found in the directory")
    items.sort(key=natural_key)
    return items


async def prepare_image_bytes(image_path: Path, max_width: Optional[int]) -> Tuple[bytes, str]:
    def _inner() -> Tuple[bytes, str]:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            if max_width and img.width > max_width:
                ratio = max_width / img.width
                new_height = max(1, int(img.height * ratio))
                img = img.resize((max_width, new_height), RESAMPLE_LANCZOS)
            ext = image_path.suffix.lower()
            fmt = "PNG" if ext == ".png" else "JPEG"
            buffer = BytesIO()
            if fmt == "JPEG":
                img = img.convert("RGB")
                img.save(buffer, format=fmt, quality=90)
                fmt_token = "jpeg"
            else:
                img.save(buffer, format=fmt)
                fmt_token = "png"
            return buffer.getvalue(), fmt_token

    return await asyncio.to_thread(_inner)


def build_payload_variants(model: str, image_bytes: bytes, image_format: str) -> List[Tuple[str, str, dict]]:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:image/{image_format};base64,{encoded}"
    variants: List[Tuple[str, str, dict]] = []

    variants.append(
        (
            "image+content",
            OCR_ENDPOINT,
            {
                "model": model,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"image": {"format": image_format, "content": encoded}},
                                {"text": OCR_PROMPT},
                            ],
                        }
                    ]
                },
            },
        )
    )

    variants.append(
        (
            "typed-image",
            OCR_ENDPOINT,
            {
                "model": model,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": {"format": image_format, "content": encoded},
                                },
                                {"type": "text", "text": OCR_PROMPT},
                            ],
                        }
                    ]
                },
            },
        )
    )

    variants.append(
        (
            "data-url",
            OCR_ENDPOINT,
            {
                "model": model,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_image", "image_url": data_url},
                                {"type": "text", "text": OCR_PROMPT},
                            ],
                        }
                    ]
                },
            },
        )
    )

    variants.append(
        (
            "compatible-image-url",
            COMPATIBLE_ENDPOINT,
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                            {
                                "type": "text",
                                "text": OCR_PROMPT,
                            },
                        ],
                    }
                ],
                "temperature": 0,
            },
        )
    )

    variants.append(
        (
            "compatible-b64-json",
            COMPATIBLE_ENDPOINT,
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image": {
                                    "b64_json": encoded,
                                    "media_type": f"image/{image_format}",
                                },
                            },
                            {
                                "type": "text",
                                "text": OCR_PROMPT,
                            },
                        ],
                    }
                ],
                "temperature": 0,
            },
        )
    )

    variants.append(
        (
            "compatible-image-base64",
            COMPATIBLE_ENDPOINT,
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_base64",
                                "image_base64": encoded,
                                "image_format": image_format,
                            },
                            {
                                "type": "text",
                                "text": OCR_PROMPT,
                            },
                        ],
                    }
                ],
                "temperature": 0,
            },
        )
    )

    return variants


def extract_text_from_response(data: dict) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        choice0 = choices[0] if isinstance(choices[0], dict) else {}
        message = choice0.get("message", {}) if isinstance(choice0, dict) else {}
        content = message.get("content")
        if isinstance(content, list):
            fragments: List[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type in {"text", "output_text"} and item.get("text"):
                    fragments.append(str(item["text"]))
                elif item_type is None and item.get("text"):
                    fragments.append(str(item["text"]))
            joined = "\n".join(fragments).strip()
            if joined:
                return joined
        if isinstance(content, str) and content.strip():
            return content.strip()
        text_value = message.get("text")
        if isinstance(text_value, str) and text_value.strip():
            return text_value.strip()
        if "output_text" in message and isinstance(message["output_text"], str):
            candidate = message["output_text"].strip()
            if candidate:
                return candidate

    output_text = data.get("output_text")
    if isinstance(output_text, list):
        joined = "\n".join(str(x) for x in output_text if x)
        if joined.strip():
            return joined.strip()
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = data.get("output") or {}
    if isinstance(output, dict):
        choices = output.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            content = message.get("content")
            if isinstance(content, list):
                texts = [c.get("text") for c in content if isinstance(c, dict) and c.get("text")]
                joined = "\n".join(t for t in texts if t)
                if joined.strip():
                    return joined.strip()
            text_value = message.get("text")
            if isinstance(text_value, str) and text_value.strip():
                return text_value.strip()
        text_output = output.get("text")
        if isinstance(text_output, list):
            joined = "\n".join(str(item) for item in text_output if item)
            if joined.strip():
                return joined.strip()
        if isinstance(text_output, str) and text_output.strip():
            return text_output.strip()
    if "text" in data and isinstance(data["text"], str) and data["text"].strip():
        return data["text"].strip()
    raise ValueError("Could not extract text content from model response")


async def call_ocr_with_retry(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    image_bytes: bytes,
    image_format: str,
) -> str:
    payload_variants = build_payload_variants(model, image_bytes, image_format)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        for variant_name, endpoint, payload in payload_variants:
            try:
                response = await client.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                if isinstance(data, dict):
                    if data.get("code") and data.get("code") not in {"Success", "OK", 200}:
                        raise RuntimeError(f"DashScope returned an error: {data.get('code')} {data.get('message')}")
                text = extract_text_from_response(data)
                await asyncio.sleep(random.uniform(*RATE_LIMIT_RANGE))
                return text
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status == 400 and exc.response.text and "url error" in exc.response.text.lower():
                    last_error = exc
                    continue
                if status >= 400 and status < 500 and status != 429:
                    raise RuntimeError(f"Request failed (HTTP {status}): {exc.response.text}") from exc
                last_error = exc
            except (httpx.RequestError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
                last_error = exc
        delay = BACKOFF_BASE * (2 ** (attempt - 1))
        delay += random.uniform(0, BACKOFF_JITTER)
        await asyncio.sleep(delay)
    raise RuntimeError(f"OCR failed after multiple attempts: {last_error}")


def post_clean_markdown(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cleaned: List[str] = []
    buffer: List[str] = []
    in_code_block = False

    def flush_buffer() -> None:
        if buffer:
            merged = " ".join(buffer).strip()
            if merged:
                cleaned.append(merged)
            buffer.clear()

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_buffer()
            cleaned.append(line)
            in_code_block = not in_code_block
            continue
        if in_code_block:
            cleaned.append(line)
            continue
        if not stripped:
            flush_buffer()
            if cleaned and cleaned[-1] == "":
                continue
            cleaned.append("")
            continue
        if any(pattern.match(stripped) for pattern in HEADER_PATTERNS):
            continue
        if re.match(r"^[-*+]\s+", stripped) or re.match(r"^\d+\.\s+", stripped):
            flush_buffer()
            cleaned.append(line)
            continue
        if stripped.startswith(("#", ">")):
            flush_buffer()
            cleaned.append(stripped)
            continue
        if "|" in stripped and stripped.count("|") >= 2:
            flush_buffer()
            cleaned.append(stripped)
            continue
        buffer.append(stripped)

    flush_buffer()

    result: List[str] = []
    last_blank = False
    for line in cleaned:
        if line == "":
            if not last_blank:
                result.append("")
            last_blank = True
        else:
            result.append(line)
            last_blank = False

    return "\n".join(result).strip()


def write_page_md(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = content.strip()
    if text:
        text += "\n"
    else:
        text = "\n"
    path.write_text(text, encoding="utf-8")


def merge_markdown(pages: Sequence[PageTask], output_path: Path) -> List[PageTask]:
    available: List[PageTask] = []
    blocks: List[str] = []
    for page in pages:
        if not page.md_path.exists():
            continue
        content = page.md_path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        if blocks and blocks[-1] != "":
            blocks.append("")
        blocks.append(content)
        blocks.append("")
        available.append(page)
    if not available:
        raise RuntimeError("No Markdown pages are available for merging")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged = "\n".join(blocks).strip() + "\n"
    output_path.write_text(merged, encoding="utf-8")
    return available


def build_epub(
    pandoc_path: str,
    book_md: Path,
    epub_path: Path,
    title: str,
    author: str,
    lang: str,
    cover: Optional[Path],
) -> None:
    cmd = [
        pandoc_path,
        str(book_md),
        "-o",
        str(epub_path),
        "--toc",
        "--toc-depth=3",
        f"--metadata=title={title}",
        f"--metadata=creator={author}",
        f"--metadata=lang={lang}",
    ]
    if cover:
        cmd.append(f"--epub-cover-image={cover}")
    subprocess.run(cmd, check=True)


def convert_with_calibre(epub_path: Path, target_suffix: str) -> Path:
    output_path = epub_path.with_suffix(target_suffix)
    cmd = ["ebook-convert", str(epub_path), str(output_path)]
    subprocess.run(cmd, check=True)
    return output_path


def detect_pandoc() -> str:
    path = shutil.which("pandoc")
    if not path:
        raise FileNotFoundError("pandoc not found. Please install it before running this script.")
    return path


def detect_calibre() -> Optional[str]:
    return shutil.which("ebook-convert")


async def process_page(
    page: PageTask,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    api_key: str,
    args: argparse.Namespace,
) -> PageResult:
    if args.dry_run:
        logger.info("[DRY RUN] Page %d: %s -> %s", page.page_number, page.image_path.name, page.md_path)
        return PageResult(page.page_number, "dry")
    if args.skip_ocr_existing and page.md_path.exists():
        logger.debug("Skipping existing Markdown for page %d", page.page_number)
        return PageResult(page.page_number, "skipped")
    async with semaphore:
        try:
            image_bytes, image_format = await prepare_image_bytes(page.image_path, args.max_width)
            text = await call_ocr_with_retry(client, api_key, args.model, image_bytes, image_format)
            cleaned = post_clean_markdown(text)
            await asyncio.to_thread(write_page_md, page.md_path, cleaned)
            return PageResult(page.page_number, "success")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to process page %d: %s", page.page_number, exc)
            return PageResult(page.page_number, "failed", str(exc))


async def process_all_pages(
    pages: Sequence[PageTask],
    api_key: str,
    args: argparse.Namespace,
) -> List[PageResult]:
    if not pages:
        return []
    timeout = httpx.Timeout(120.0, connect=30.0)
    limits = httpx.Limits(max_keepalive_connections=args.concurrency, max_connections=args.concurrency)
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        tasks = [process_page(page, client, semaphore, api_key, args) for page in pages]
        results: List[PageResult] = []
        with tqdm(total=len(tasks), desc="OCR pages") as progress:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                progress.update(1)
        return results


def plan_outputs(images: Sequence[Path], out_root: Path, out_name: str) -> List[PageTask]:
    tasks: List[PageTask] = []
    for idx, image in enumerate(images, start=1):
        md_path = out_root / "pages" / f"page-{idx:04d}.md"
        tasks.append(PageTask(index=idx - 1, page_number=idx, image_path=image, md_path=md_path))
    return tasks


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR scanned pages to Markdown and export EPUB/AZW3/MOBI.")
    parser.add_argument("--images-dir", required=True, help="Directory containing page images")
    parser.add_argument("--title", default="Untitled Book", help="Book title")
    parser.add_argument("--author", default="Unknown Author", help="Author name")
    parser.add_argument("--lang", default="zh-CN", help="Language code (default zh-CN)")
    parser.add_argument("--max-width", type=int, help="Maximum width before upload (pixels)")
    parser.add_argument("--concurrency", type=int, default=4, help="Async concurrency for OCR requests")
    parser.add_argument("--model", default="qwen2.5-vl-7b-instruct", help="DashScope model name")
    parser.add_argument("--cover", help="Cover image path for EPUB metadata")
    parser.add_argument("--out-name", default="book", help="Output filename prefix")
    parser.add_argument("--skip-ocr-existing", action="store_true", help="Skip pages that already have Markdown")
    parser.add_argument("--to-azw3", dest="to_azw3", action="store_true", help="Generate AZW3 (requires Calibre)")
    parser.add_argument("--to-mobi", dest="to_mobi", action="store_true", help="Generate MOBI (requires Calibre)")
    parser.add_argument("--from-list", help="Text file listing images in processing order")
    parser.add_argument("--pages", help="Process specific page numbers, e.g., 1-10,15,20-25")
    parser.add_argument("--dry-run", action="store_true", help="Only show planned pages without OCR")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def select_pages(all_images: List[Path], pages_spec: Optional[str]) -> List[Path]:
    if not pages_spec:
        return all_images
    indices = parse_pages_spec(pages_spec, len(all_images))
    lookup = {idx + 1: path for idx, path in enumerate(all_images)}
    selected = [lookup[i] for i in indices if i in lookup]
    return selected


def resolve_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    return Path(value).expanduser().resolve()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_arguments(argv)
    setup_logging(args.verbose)

    images_dir = Path(args.images_dir).expanduser().resolve()
    from_list_path = resolve_path(args.from_list)
    cover_path = resolve_path(args.cover)

    try:
        images = gather_images(images_dir, from_list_path)
        images = select_pages(images, args.pages)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to gather images: %s", exc)
        return 2

    if not images:
        logger.error("No pages to process.")
        return 2

    out_root = images_dir / "_out"
    tasks = plan_outputs(images, out_root, args.out_name)

    need_ocr = not args.dry_run
    if need_ocr:
        need_ocr = False
        for task in tasks:
            if args.skip_ocr_existing and task.md_path.exists():
                continue
            need_ocr = True
            break
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if need_ocr and not api_key:
        logger.error("Environment variable DASHSCOPE_API_KEY is not set.")
        return 2

    if args.dry_run:
        for task in tasks:
            logger.info("DRY: Page %d -> %s", task.page_number, task.image_path)
        return 0

    try:
        results = asyncio.run(process_all_pages(tasks, api_key or "", args))
    except Exception as exc:  # noqa: BLE001
        logger.error("Fatal error while processing pages: %s", exc)
        return 2

    success_pages = [r for r in results if r.status == "success"]
    skipped_pages = [r for r in results if r.status == "skipped"]
    failed_pages = [r for r in results if r.status == "failed"]

    logger.info("Completed %d pages, skipped %d, failed %d.", len(success_pages), len(skipped_pages), len(failed_pages))
    if failed_pages:
        logger.warning("Pages that failed: %s", ", ".join(str(r.page_number) for r in failed_pages))

    available_md_tasks = [task for task in tasks if task.md_path.exists()]
    if not available_md_tasks:
        logger.error("No Markdown pages were produced; aborting.")
        return 2

    book_md_path = out_root / f"{args.out_name}.md"
    try:
        merge_markdown(tasks, book_md_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to merge Markdown pages: %s", exc)
        return 2

    try:
        pandoc_path = detect_pandoc()
    except Exception as exc:  # noqa: BLE001
        logger.error("%s", exc)
        return 2

    epub_path = out_root / f"{args.out_name}.epub"
    try:
        build_epub(pandoc_path, book_md_path, epub_path, args.title, args.author, args.lang, cover_path)
    except subprocess.CalledProcessError as exc:
        logger.error("pandoc failed while building the EPUB: %s", exc)
        return 2

    calibre_path = detect_calibre()
    if args.to_azw3 or args.to_mobi:
        if not calibre_path:
            logger.warning("ebook-convert not found; skipping AZW3/MOBI outputs.")
        else:
            if args.to_azw3:
                try:
                    convert_with_calibre(epub_path, ".azw3")
                    logger.info("Created AZW3: %s", epub_path.with_suffix(".azw3"))
                except subprocess.CalledProcessError as exc:
                    logger.error("Failed to create AZW3: %s", exc)
            if args.to_mobi:
                try:
                    convert_with_calibre(epub_path, ".mobi")
                    logger.info("Created MOBI: %s", epub_path.with_suffix(".mobi"))
                except subprocess.CalledProcessError as exc:
                    logger.error("Failed to create MOBI: %s", exc)

    logger.info("Wrote merged Markdown to %s", book_md_path)
    logger.info("Wrote EPUB to %s", epub_path)

    if failed_pages and not success_pages and not skipped_pages:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
