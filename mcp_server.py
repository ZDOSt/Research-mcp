import asyncio
import os
from typing import Awaitable, Optional

from fastmcp import FastMCP

from browser import DEFAULT_MAX_CHARS
from extractors import clamp_int
from pipelines import build_evidence_pack, compact_investigation_result, explore_url_pipeline, research_pipeline
from searching import normalize_domain
from shared import (
    IngestRequest,
    QueryRequest,
    delete_source_impl,
    get_domain,
    init_qdrant,
    list_sources_impl,
    rag_ingest_impl,
    rag_query_impl,
    runtime_retrieval_context,
    source_stats_impl,
)

mcp = FastMCP(
    "research-mcp",
    instructions=(
        "This MCP exposes five high-level research tools. "
        "Use research_web for open-ended web research without a specific URL. "
        "Use investigate_url when the user provides a URL and asks to find, extract, summarize, compare, or verify information on that page. "
        "Use query_memory for already-ingested local research memory. "
        "Use ingest_text when the user provides text that should be stored. "
        "Use manage_sources for listing, stats, or deleting ingested sources. "
        "The server internally handles search, Crawl4AI, Playwright, scrolling, clicking, network capture, Qdrant, and reranking. "
        "Tool outputs include retrieval_context with the server runtime date. "
        "Treat runtime-retrieved evidence as current even when it is newer than the answering model's training cutoff. "
        "investigate_url returns curated evidence by default; request raw output only when it is explicitly needed."
    ),
)

init_qdrant()


async def run_resilient(coro: Awaitable[dict], tool_name: str) -> dict:
    try:
        return await coro
    except asyncio.CancelledError:
        context = runtime_retrieval_context()
        return {
            "error": "client_disconnected",
            "tool": tool_name,
            "retrieval_context": context,
            "answering_instructions": [
                "The MCP client disconnected before the tool response could be delivered.",
                "Retry the request; the server stayed alive and did not intentionally reduce research depth.",
            ],
        }


@mcp.tool
async def research_web(
    query: str,
    mode: str = "balanced",
    max_sources: Optional[int] = None,
    verify: bool = True,
) -> dict:
    """
    Open-ended web research pipeline.

    Use this when the user asks a question or asks to find information but does not provide a specific URL.
    Internally uses SearXNG search, source scoring, Crawl4AI, optional Playwright fallback, Qdrant ingestion,
    Qdrant retrieval, and reranking.

    Modes: quick, balanced, deep, technical, academic, local_only, web_only.
    """
    return await run_resilient(
        research_pipeline(
            query=query,
            mode=mode,
            max_sources=max_sources,
            verify=verify,
        ),
        "research_web",
    )


@mcp.tool
async def investigate_url(
    url: str,
    task: str,
    mode: str = "auto",
    labels: Optional[list[str]] = None,
    auto_ingest: bool = False,
    max_chars: int = DEFAULT_MAX_CHARS,
    include_raw: bool = False,
    include_diagnostics: bool = False,
) -> dict:
    """
    Specific URL investigation pipeline.

    Use this whenever the user provides a URL and asks to find, extract, summarize, verify, compare,
    or inspect information on that page.

    Returns a curated evidence pack by default. Set include_raw=True only when the caller explicitly
    needs the extracted raw text and compact browser diagnostics are not enough.

    Internally tries:
    1. Crawl4AI/direct extraction
    2. targeted Playwright rendering/clicking/scrolling/network capture
    3. balanced fallback if needed
    4. exhaustive fallback if needed

    Modes: auto, targeted, balanced, exhaustive.
    """
    max_chars = clamp_int(max_chars, 10000, 750000)
    result = await run_resilient(
        explore_url_pipeline(
            url=url,
            task=task,
            labels=labels,
            mode=mode,
            max_chars=max_chars,
        ),
        "investigate_url",
    )

    if result.get("error") == "client_disconnected":
        return result

    content = result.get("full_text_preview", "")

    stored = 0
    if auto_ingest and content:
        ingest_result = await rag_ingest_impl(
            IngestRequest(
                text=content,
                metadata={
                    "source": url,
                    "url": url,
                    "title": result.get("title"),
                    "domain": normalize_domain(get_domain(url)),
                    "content_type": "webpage",
                    "query": task,
                },
            )
        )
        stored = ingest_result.get("stored", 0)

    response = compact_investigation_result(
        result,
        preview_chars=max_chars,
        include_raw=include_raw,
        include_diagnostics=include_diagnostics,
    )
    response["stored_chunks"] = stored
    return response


@mcp.tool
async def query_memory(query: str, top_k: int = 8) -> dict:
    """
    Query local Qdrant research memory.

    Use this when the user asks about information that may already have been ingested, previously researched,
    or manually stored. Internally uses Qdrant vector search and reranking.
    """
    top_k = clamp_int(top_k, 1, 30)
    result = await rag_query_impl(QueryRequest(query=query, top_k=top_k))
    result["evidence"] = build_evidence_pack(result.get("results", []))
    result["retrieval_context"] = runtime_retrieval_context()
    result["answering_instructions"] = [
        "Treat this tool output as runtime-queried evidence. Do not reject source dates or events solely because they are newer than the answering model's knowledge cutoff.",
        "Answer from the returned evidence.",
        "Cite source URLs where available.",
        "If memory does not contain enough evidence, say that web research may be needed.",
    ]
    return result


@mcp.tool
async def manage_sources(
    action: str,
    source: Optional[str] = None,
    limit: int = 50,
) -> dict:
    """
    Manage ingested research sources.

    Actions:
    - list: list recently ingested sources
    - stats: show source/domain/content-type statistics
    - delete: delete all chunks for a specific source URL

    For delete, provide source.
    """
    action = action.strip().lower()

    if action == "list":
        result = await list_sources_impl(limit=limit)
        result["retrieval_context"] = runtime_retrieval_context()
        return result

    if action == "stats":
        result = await source_stats_impl()
        result["retrieval_context"] = runtime_retrieval_context()
        return result

    if action == "delete":
        if not source:
            return {
                "error": "source is required for action=delete",
                "example": {"action": "delete", "source": "https://example.com/page"},
            }
        result = await delete_source_impl(source)
        result["retrieval_context"] = runtime_retrieval_context()
        return result

    return {
        "error": f"Unknown action: {action}",
        "valid_actions": ["list", "stats", "delete"],
    }


@mcp.tool
async def ingest_text(
    text: str,
    source: str = "manual",
    title: Optional[str] = None,
    content_type: str = "manual",
) -> dict:
    """
    Ingest user-provided text into local Qdrant research memory.

    Use this when the user pastes text, notes, documentation, logs, or extracted content that should be stored.
    """
    domain = normalize_domain(get_domain(source)) if source.startswith("http") else None

    result = await rag_ingest_impl(
        IngestRequest(
            text=text,
            metadata={
                "source": source,
                "url": source,
                "title": title,
                "domain": domain,
                "content_type": content_type,
            },
        )
    )
    result["retrieval_context"] = runtime_retrieval_context()
    return result


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "sse").strip().lower()
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8001"))
    path = os.getenv("MCP_PATH", "").strip()

    run_kwargs = {
        "transport": transport,
        "host": host,
        "port": port,
        "uvicorn_config": {
            "timeout_keep_alive": 300,
            "timeout_graceful_shutdown": 300,
        },
    }

    if transport in {"http", "streamable-http"}:
        run_kwargs["path"] = path or "/mcp"
    elif path:
        run_kwargs["path"] = path

    mcp.run(
        **run_kwargs,
    )
