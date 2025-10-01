#!/usr/bin/env python3
"""\
用法示例：
    python ocr_md_book.py \
        --images-dir ./book_images \
        --title "我的扫描书" \
        --author "未知作者"

该工具会将扫描页图像 OCR 成 Markdown，并合并为电子书（EPUB，及可选的 AZW3/MOBI）。
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
你是“图片转可读电子书”的排版专家。请将图片中的内容转写为 GitHub-Flavored Markdown (GFM)，严格遵循：
1) 不要臆造任何不存在的内容；保证阅读顺序正确。
2) 标题用 # / ## / ### 表示层级；勿把正文误判为标题。
3) 强调用 **加粗** 与 *斜体* 表达。
4) 列表使用 - 和 1. 2. 3.，子级每层缩进两个空格。
5) 表格用标准 Markdown 表格语法（| th | th |）。
6) 代码与公式保留为 ``` 代码块 或 `行内`，数学公式保留 $...$ / $$...$$。
7) 去除页眉/页脚/页码/水印/扫描噪声。
8) 段落之间空一行；不要用硬换行截断正常句子。
9) 图片中的图题/脚注以 *斜体* 或普通段落保留。
仅输出 Markdown 纯文本，不要返回 JSON 或解释。
"""
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
HEADER_PATTERNS = [
    re.compile(r"^第?\s*\d+\s*[页頁]$", re.IGNORECASE),
    re.compile(r"^page\s*\d+$", re.IGNORECASE),
    re.compile(r"^\d{1,4}$"),
    re.compile(r"^(confidential|draft|company\s+name|proprietary).*$", re.IGNORECASE),
    re.compile(r"^版权所有.*$"),
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
                raise FileNotFoundError(f"列表中的文件不存在：{candidate}")
            if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            items.append(candidate)
        if not items:
            raise ValueError("指定的列表中没有有效的图像文件")
        return items

    if not images_dir.exists():
        raise FileNotFoundError(f"图像目录不存在：{images_dir}")
    items = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not items:
        raise ValueError("图像目录中未找到 PNG/JPG 图片")
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
    raise ValueError("未能从模型响应中解析文本内容")


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
                        raise RuntimeError(f"DashScope 返回错误：{data.get('code')} {data.get('message')}")
                text = extract_text_from_response(data)
                await asyncio.sleep(random.uniform(*RATE_LIMIT_RANGE))
                return text
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status == 400 and exc.response.text and "url error" in exc.response.text.lower():
                    last_error = exc
                    continue
                if status >= 400 and status < 500 and status != 429:
                    raise RuntimeError(f"请求失败（HTTP {status}）：{exc.response.text}") from exc
                last_error = exc
            except (httpx.RequestError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
                last_error = exc
        delay = BACKOFF_BASE * (2 ** (attempt - 1))
        delay += random.uniform(0, BACKOFF_JITTER)
        await asyncio.sleep(delay)
    raise RuntimeError(f"多次尝试后仍无法完成 OCR：{last_error}")


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
        raise RuntimeError("没有可用的 Markdown 页，无法合并")
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
        raise FileNotFoundError("未找到 pandoc，请先安装后再运行。")
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
        logger.info("[DRY RUN] 第 %d 页：%s -> %s", page.page_number, page.image_path.name, page.md_path)
        return PageResult(page.page_number, "dry")
    if args.skip_ocr_existing and page.md_path.exists():
        logger.debug("跳过已存在的 Markdown：第 %d 页", page.page_number)
        return PageResult(page.page_number, "skipped")
    async with semaphore:
        try:
            image_bytes, image_format = await prepare_image_bytes(page.image_path, args.max_width)
            text = await call_ocr_with_retry(client, api_key, args.model, image_bytes, image_format)
            cleaned = post_clean_markdown(text)
            await asyncio.to_thread(write_page_md, page.md_path, cleaned)
            return PageResult(page.page_number, "success")
        except Exception as exc:  # noqa: BLE001
            logger.error("第 %d 页处理失败：%s", page.page_number, exc)
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
        with tqdm(total=len(tasks), desc="OCR 页") as progress:
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
    parser = argparse.ArgumentParser(description="将扫描页 OCR 为 Markdown 并导出 EPUB/AZW3/MOBI。")
    parser.add_argument("--images-dir", required=True, help="包含页图像的目录")
    parser.add_argument("--title", default="未命名书籍", help="电子书标题")
    parser.add_argument("--author", default="未知作者", help="作者")
    parser.add_argument("--lang", default="zh-CN", help="语言代码（默认 zh-CN）")
    parser.add_argument("--max-width", type=int, help="上传前的最大宽度（像素）")
    parser.add_argument("--concurrency", type=int, default=4, help="异步并发数")
    parser.add_argument("--model", default="qwen2.5-vl-7b-instruct", help="DashScope 模型名称")
    parser.add_argument("--cover", help="用于 EPUB 元数据的封面图路径")
    parser.add_argument("--out-name", default="book", help="输出文件前缀名")
    parser.add_argument("--skip-ocr-existing", action="store_true", help="若页面 Markdown 已存在则跳过")
    parser.add_argument("--to-azw3", dest="to_azw3", action="store_true", help="生成 AZW3（需 Calibre）")
    parser.add_argument("--to-mobi", dest="to_mobi", action="store_true", help="生成 MOBI（需 Calibre）")
    parser.add_argument("--from-list", help="按行指定图像路径列表")
    parser.add_argument("--pages", help="页码筛选，例如 1-10,15,20-25")
    parser.add_argument("--dry-run", action="store_true", help="仅列出将处理的页面，不执行 OCR")
    parser.add_argument("--verbose", action="store_true", help="打印调试信息")
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
        logger.error("收集图像失败：%s", exc)
        return 2

    if not images:
        logger.error("没有可处理的页面。")
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
        logger.error("未设置环境变量 DASHSCOPE_API_KEY。")
        return 2

    if args.dry_run:
        for task in tasks:
            logger.info("DRY: 第 %d 页 -> %s", task.page_number, task.image_path)
        return 0

    try:
        results = asyncio.run(process_all_pages(tasks, api_key or "", args))
    except Exception as exc:  # noqa: BLE001
        logger.error("处理过程中出现严重错误：%s", exc)
        return 2

    success_pages = [r for r in results if r.status == "success"]
    skipped_pages = [r for r in results if r.status == "skipped"]
    failed_pages = [r for r in results if r.status == "failed"]

    logger.info("成功 %d 页，跳过 %d 页，失败 %d 页。", len(success_pages), len(skipped_pages), len(failed_pages))
    if failed_pages:
        logger.warning("失败页码：%s", ", ".join(str(r.page_number) for r in failed_pages))

    available_md_tasks = [task for task in tasks if task.md_path.exists()]
    if not available_md_tasks:
        logger.error("没有任何 Markdown 输出，终止。")
        return 2

    book_md_path = out_root / f"{args.out_name}.md"
    try:
        merge_markdown(tasks, book_md_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("合并 Markdown 失败：%s", exc)
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
        logger.error("pandoc 构建 EPUB 失败：%s", exc)
        return 2

    calibre_path = detect_calibre()
    if args.to_azw3 or args.to_mobi:
        if not calibre_path:
            logger.warning("未检测到 ebook-convert，跳过 AZW3/MOBI 输出。")
        else:
            if args.to_azw3:
                try:
                    convert_with_calibre(epub_path, ".azw3")
                    logger.info("已生成 AZW3：%s", epub_path.with_suffix(".azw3"))
                except subprocess.CalledProcessError as exc:
                    logger.error("生成 AZW3 失败：%s", exc)
            if args.to_mobi:
                try:
                    convert_with_calibre(epub_path, ".mobi")
                    logger.info("已生成 MOBI：%s", epub_path.with_suffix(".mobi"))
                except subprocess.CalledProcessError as exc:
                    logger.error("生成 MOBI 失败：%s", exc)

    logger.info("已生成 Markdown：%s", book_md_path)
    logger.info("已生成 EPUB：%s", epub_path)

    if failed_pages and not success_pages and not skipped_pages:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
