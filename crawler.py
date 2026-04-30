import json
from typing import Any

import httpx

from extractors import extract_title_from_html, html_to_text, parse_maybe_json_text
from shared import CRAWL4AI_URL

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


async def crawl4ai_request(payload: dict, timeout: float = 180.0) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{CRAWL4AI_URL}/crawl", json=payload)
        resp.raise_for_status()
        return resp.json()


async def crawl4ai_markdown_request(url: str, timeout: float = 120.0) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{CRAWL4AI_URL}/md", json={"url": url, "f": "fit", "c": "0"})
        resp.raise_for_status()
        try:
            data = resp.json()
        except Exception:
            data = {"url": url, "markdown": resp.text, "success": True}

    markdown = data.get("markdown") or ""
    return {
        "url": data.get("url") or url,
        "markdown": markdown,
        "content": markdown,
        "title": None,
        "success": bool(data.get("success", True)),
        "extraction_method": "crawl4ai_md",
    }


def first_crawl4ai_result(data: dict) -> dict:
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        for item in data["results"]:
            if isinstance(item, dict):
                item = dict(item)
                item["_crawl4ai_success"] = data.get("success")
                item["_crawl4ai_server_processing_time_s"] = data.get("server_processing_time_s")
                return item
        return {}

    return data if isinstance(data, dict) else {}


def extract_markdown(value: Any) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        for key in ("fit_markdown", "raw_markdown", "markdown_with_citations", "markdown"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return text

    return ""


def extract_content(crawl_data: dict) -> str:
    content = crawl_data.get("content") or crawl_data.get("cleaned_text") or ""

    if not content:
        content = extract_markdown(crawl_data.get("markdown"))

    if not content and crawl_data.get("extracted_content"):
        extracted = crawl_data.get("extracted_content")
        content = parse_maybe_json_text(extracted) if isinstance(extracted, str) else json.dumps(extracted)

    if not content and crawl_data.get("html"):
        content = html_to_text(crawl_data.get("html") or "")

    return (content or "").strip()


def extract_title(crawl_data: dict, fallback: str | None = None) -> str | None:
    title = crawl_data.get("title")
    if title:
        return title

    metadata = crawl_data.get("metadata")
    if isinstance(metadata, dict):
        title = metadata.get("title")
        if title:
            return title

    return fallback


async def direct_fetch_url(url: str) -> dict:
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True, headers=BROWSER_HEADERS) as client:
        resp = await client.get(url)
        resp.raise_for_status()

    raw_body = resp.text
    content_type = resp.headers.get("content-type", "")
    title = extract_title_from_html(raw_body)

    if "json" in content_type.lower() or raw_body.strip().startswith(("{", "[")):
        text = parse_maybe_json_text(raw_body)
    else:
        text = html_to_text(raw_body)

    return {
        "url": str(resp.url),
        "status_code": resp.status_code,
        "content_type": content_type,
        "title": title,
        "content": text,
        "markdown": text,
        "raw_html_chars": len(raw_body),
        "extraction_method": "direct_http_fallback",
    }


def crawl4ai_payload(url: str, crawler_config: dict | None = None) -> dict:
    # Current Crawl4AI Docker API expects urls/browser_config/crawler_config.
    return {
        "urls": [url],
        "browser_config": {},
        "crawler_config": crawler_config or {},
    }


async def crawl_url_impl(url: str, config: dict | None = None) -> dict:
    attempts = [
        crawl4ai_payload(url, config or {}),
    ]

    errors = []

    for payload in attempts:
        try:
            data = first_crawl4ai_result(await crawl4ai_request(payload))
            content = extract_content(data)
            if content and len(content) >= 200:
                data["extraction_method"] = "crawl4ai"
                return data

            errors.append("Crawl4AI returned too little content")
        except Exception as exc:
            errors.append(str(exc))

    try:
        data = await crawl4ai_markdown_request(url)
        content = extract_content(data)
        if content and len(content) >= 200:
            data["crawl4ai_errors"] = errors
            return data
        errors.append("Crawl4AI /md returned too little content")
    except Exception as exc:
        errors.append(f"crawl4ai /md failed: {exc}")

    try:
        fallback = await direct_fetch_url(url)
        fallback["crawl4ai_errors"] = errors
        return fallback
    except Exception as exc:
        errors.append(f"direct fallback failed: {exc}")
        raise RuntimeError("; ".join(errors))
