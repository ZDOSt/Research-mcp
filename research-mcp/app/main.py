from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import httpx
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount


SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "http://searxng:8080")
CRAWL4AI_BASE_URL = os.getenv("CRAWL4AI_BASE_URL", "http://crawl4ai:11235")
RERANKER_BASE_URL = os.getenv("RERANKER_BASE_URL", "http://reranker:8010")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))

mcp = FastMCP("research-mcp")


async def _request_json(
    method: str,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_payload: Optional[Dict[str, Any]] = None,
) -> Any:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.request(method, url, params=params, json=json_payload)
        response.raise_for_status()
        ctype = response.headers.get("content-type", "")
        if "application/json" in ctype:
            return response.json()
        text = response.text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"text": text}


@mcp.tool()
async def search_web(
    query: str,
    categories: Optional[str] = None,
    engines: Optional[str] = None,
    language: Optional[str] = None,
    time_range: Optional[str] = None,
    pageno: int = 1,
    safesearch: int = 0,
) -> Dict[str, Any]:
    """Search the web through SearXNG and return JSON results."""
    return await _request_json(
        "GET",
        f"{SEARXNG_BASE_URL}/search",
        params={
            "q": query,
            "format": "json",
            "categories": categories,
            "engines": engines,
            "language": language,
            "time_range": time_range,
            "pageno": pageno,
            "safesearch": safesearch,
        },
    )


@mcp.tool()
async def rerank_results(
    query: str,
    documents: List[str],
    top_n: int = 5,
    model: str = "BAAI/bge-reranker-v2-m3",
    return_documents: bool = False,
) -> Dict[str, Any]:
    """Rerank candidate texts using the local Jina-compatible reranker."""
    return await _request_json(
        "POST",
        f"{RERANKER_BASE_URL}/v1/rerank",
        json_payload={
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": return_documents,
        },
    )


@mcp.tool()
async def crawl_markdown(
    url: str,
    filter: str = "fit",
    query: Optional[str] = None,
    cache: str = "0",
    provider: Optional[str] = None,
    temperature: Optional[float] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a URL to markdown through Crawl4AI's /md endpoint."""
    return await _request_json(
        "POST",
        f"{CRAWL4AI_BASE_URL}/md",
        json_payload={
            "url": url,
            "f": filter,
            "q": query,
            "c": cache,
            "provider": provider,
            "temperature": temperature,
            "base_url": base_url,
        },
    )


@mcp.tool()
async def crawl_html(url: str) -> Dict[str, Any]:
    """Return preprocessed HTML for a URL through Crawl4AI's /html endpoint."""
    return await _request_json(
        "POST",
        f"{CRAWL4AI_BASE_URL}/html",
        json_payload={"url": url},
    )


@mcp.tool()
async def execute_js(url: str, scripts: List[str]) -> Dict[str, Any]:
    """Execute ordered JavaScript snippets in the page context and return CrawlResult JSON."""
    return await _request_json(
        "POST",
        f"{CRAWL4AI_BASE_URL}/execute_js",
        json_payload={"url": url, "scripts": scripts},
    )


@mcp.tool()
async def screenshot_page(
    url: str,
    screenshot_wait_for: float = 2.0,
    wait_for_images: bool = False,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Capture a screenshot through Crawl4AI."""
    return await _request_json(
        "POST",
        f"{CRAWL4AI_BASE_URL}/screenshot",
        json_payload={
            "url": url,
            "screenshot_wait_for": screenshot_wait_for,
            "wait_for_images": wait_for_images,
            "output_path": output_path,
        },
    )


@mcp.tool()
async def export_pdf(url: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """Generate a PDF through Crawl4AI."""
    return await _request_json(
        "POST",
        f"{CRAWL4AI_BASE_URL}/pdf",
        json_payload={"url": url, "output_path": output_path},
    )


@mcp.tool()
async def crawl_urls(
    urls: List[str],
    browser_config: Dict[str, Any] | None = None,
    crawler_config: Dict[str, Any] | None = None,
    hooks: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Call Crawl4AI's /crawl endpoint with direct payload control."""
    payload: Dict[str, Any] = {
        "urls": urls,
        "browser_config": browser_config or {},
        "crawler_config": crawler_config or {},
    }
    if hooks is not None:
        payload["hooks"] = hooks
    return await _request_json("POST", f"{CRAWL4AI_BASE_URL}/crawl", json_payload=payload)


@mcp.tool()
async def ask_crawl4ai(
    context_type: str = "all",
    query: Optional[str] = None,
    score_ratio: float = 0.5,
    max_results: int = 20,
) -> Dict[str, Any]:
    """Query Crawl4AI's packaged code/doc context through /ask."""
    return await _request_json(
        "GET",
        f"{CRAWL4AI_BASE_URL}/ask",
        params={
            "context_type": context_type,
            "query": query,
            "score_ratio": score_ratio,
            "max_results": max_results,
        },
    )


@mcp.tool()
async def crawl4ai_schema() -> Dict[str, Any]:
    """Return the canonical Crawl4AI schema payload for BrowserConfig/CrawlerRunConfig."""
    return await _request_json("GET", f"{CRAWL4AI_BASE_URL}/schema")


@mcp.tool()
async def crawl4ai_advanced(
    endpoint: str,
    method: str = "POST",
    payload: Dict[str, Any] | None = None,
    query: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Raw pass-through to Crawl4AI endpoints. Use this for full advanced control.

    Important: for complex BrowserConfig/CrawlerRunConfig objects, send Crawl4AI's
    typed {"type": "ClassName", "params": {...}} shape.
    Use crawl4ai_schema() to inspect the canonical structure.
    """
    endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    return await _request_json(
        method.upper(),
        f"{CRAWL4AI_BASE_URL}{endpoint}",
        params=query or {},
        json_payload=payload or {},
    )


@mcp.tool()
async def deep_research(
    query: str,
    num_results: int = 8,
    rerank_top_n: int = 5,
    time_range: Optional[str] = None,
    engines: Optional[str] = None,
) -> Dict[str, Any]:
    """High-level workflow: search with SearXNG, rerank, then crawl top URLs."""
    search = await _request_json(
        "GET",
        f"{SEARXNG_BASE_URL}/search",
        params={
            "q": query,
            "format": "json",
            "engines": engines,
            "time_range": time_range,
        },
    )
    results = search.get("results", [])[:num_results]
    urls = [item.get("url") for item in results if item.get("url")]
    documents = [
        "\n".join(part for part in [item.get("title"), item.get("content"), item.get("url")] if part)
        for item in results
    ]

    reranked: Dict[str, Any] | None = None
    if documents:
        reranked = await _request_json(
            "POST",
            f"{RERANKER_BASE_URL}/v1/rerank",
            json_payload={
                "model": "BAAI/bge-reranker-v2-m3",
                "query": query,
                "documents": documents,
                "top_n": min(rerank_top_n, len(documents)),
            },
        )

    crawled: Any = None
    if urls:
        crawled = await _request_json(
            "POST",
            f"{CRAWL4AI_BASE_URL}/crawl",
            json_payload={
                "urls": urls[:rerank_top_n],
                "browser_config": {},
                "crawler_config": {
                    "type": "CrawlerRunConfig",
                    "params": {
                        "stream": False,
                    },
                },
            },
        )

    return {
        "search": search,
        "rerank": reranked,
        "crawl": crawled,
        "note": "Use crawl4ai_advanced for full BrowserConfig/CrawlerRunConfig passthrough and advanced options.",
    }


http_app = mcp.streamable_http_app()
app = Starlette(routes=[Mount("/mcp", app=http_app)])
