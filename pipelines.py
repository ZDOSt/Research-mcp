import asyncio
import re
import time
from typing import Dict, List, Optional

from browser import ABSOLUTE_MAX_CHARS, DEFAULT_MAX_CHARS, playwright_explore_page
from crawler import crawl_url_impl, extract_content, extract_title
from extractors import (
    clamp_int,
    estimate_confidence,
    extract_relevant_lines,
    extract_sections_from_text,
    extract_table_like_rows,
    extraction_sufficient,
    infer_page_labels,
    is_product_task,
    unique_preserve_order,
)
from searching import RESEARCH_MODE_CONFIG, normalize_domain, searxng_search
from shared import (
    IngestRequest,
    QueryRequest,
    get_domain,
    logger,
    rag_ingest_impl,
    rag_query_impl,
    runtime_retrieval_context,
)

URL_CONTENT_PREVIEW_LIMIT = 8_000
URL_EVIDENCE_CONTENT_PREVIEW_LIMIT = 2_000
URL_RELEVANT_LINE_LIMIT = 90
URL_RELEVANT_LINE_CHAR_LIMIT = 700
URL_SECTION_CHAR_LIMIT = 4_000
URL_SECTION_ITEM_LIMIT = 40
URL_TABLE_ROW_LIMIT = 300
URL_TABLE_ROW_CHAR_LIMIT = 900
URL_NETWORK_EVIDENCE_LIMIT = 4
URL_NETWORK_PREVIEW_LIMIT = 500
PRODUCT_URL_RE = re.compile(
    r"/(?:product|products|part|parts|catalog|p)/[^/?#]+",
    re.I,
)


def _truncate_text(value: object, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 15)].rstrip() + "\n...[truncated]"


def _looks_like_product_url(url: str) -> bool:
    return bool(PRODUCT_URL_RE.search(url or ""))


def _freshness_instruction() -> str:
    return (
        "Treat this tool output as runtime-retrieved evidence. Do not reject source dates "
        "or events solely because they are newer than the answering model's knowledge cutoff."
    )


def _stamp_retrieval_context(items: List[dict], context: dict) -> List[dict]:
    stamped = []
    retrieved_at = context.get("retrieved_at_utc")
    current_date = context.get("current_date_utc")

    for item in items or []:
        if not isinstance(item, dict):
            stamped.append(item)
            continue

        copy = dict(item)
        copy.setdefault("retrieval_context", context)
        copy.setdefault("retrieved_at_utc", retrieved_at)
        copy.setdefault("retrieval_current_date_utc", current_date)
        copy.setdefault("freshness", context.get("freshness"))
        stamped.append(copy)

    return stamped


def _compact_found_sections(sections: dict) -> dict:
    compact = {}

    for name, section in (sections or {}).items():
        if not isinstance(section, dict) or not section.get("found"):
            continue

        content = section.get("content") or ""
        items = section.get("items") or []
        compact[name] = {
            "found": True,
            "content": _truncate_text(content, URL_SECTION_CHAR_LIMIT),
            "items": [
                _truncate_text(item, URL_RELEVANT_LINE_CHAR_LIMIT)
                for item in items[:URL_SECTION_ITEM_LIMIT]
            ],
            "truncated": len(str(content)) > URL_SECTION_CHAR_LIMIT or len(items) > URL_SECTION_ITEM_LIMIT,
        }

    return compact


def _compact_network_responses(responses: list) -> list:
    compact = []

    for item in (responses or [])[:URL_NETWORK_EVIDENCE_LIMIT]:
        content_type = (item.get("content_type") or "").lower()
        resource_type = (item.get("resource_type") or "").lower()
        if resource_type in {"script", "stylesheet", "image", "media", "font"}:
            continue
        if "javascript" in content_type or "text/css" in content_type:
            continue

        preview = item.get("preview") or item.get("text") or ""
        if not preview:
            continue

        compact.append(
            {
                "url": item.get("url"),
                "status": item.get("status"),
                "content_type": item.get("content_type"),
                "resource_type": item.get("resource_type"),
                "text_chars": item.get("text_chars"),
                "preview": _truncate_text(preview, URL_NETWORK_PREVIEW_LIMIT),
            }
        )

    return compact


def compact_investigation_result(
    result: dict,
    preview_chars: int = URL_CONTENT_PREVIEW_LIMIT,
    include_raw: bool = False,
    include_diagnostics: bool = False,
) -> dict:
    preview_chars = clamp_int(preview_chars, 2_000, URL_CONTENT_PREVIEW_LIMIT)
    content = result.get("full_text_preview") or ""
    found_sections = _compact_found_sections(result.get("found_sections") or {})
    relevant_lines = [
        _truncate_text(line, URL_RELEVANT_LINE_CHAR_LIMIT)
        for line in (result.get("relevant_lines") or [])[:URL_RELEVANT_LINE_LIMIT]
    ]
    table_like_rows = [
        _truncate_text(row, URL_TABLE_ROW_CHAR_LIMIT)
        for row in (result.get("table_like_rows") or [])[:URL_TABLE_ROW_LIMIT]
    ]
    network_evidence = _compact_network_responses(result.get("network_responses") or [])

    evidence = []
    evidence_id = 1

    for name, section in found_sections.items():
        evidence.append(
            {
                "evidence_id": evidence_id,
                "type": "section",
                "label": name,
                "text": section.get("content", ""),
            }
        )
        evidence_id += 1

    if relevant_lines:
        evidence.append(
            {
                "evidence_id": evidence_id,
                "type": "relevant_lines",
                "lines": relevant_lines,
            }
        )
        evidence_id += 1

    if table_like_rows:
        evidence.append(
            {
                "evidence_id": evidence_id,
                "type": "table_like_rows",
                "rows": table_like_rows,
                "row_count_returned": len(table_like_rows),
                "row_count_total": result.get("table_like_row_count", len(table_like_rows)),
            }
        )
        evidence_id += 1

    for item in network_evidence:
        evidence.append(
            {
                "evidence_id": evidence_id,
                "type": "network_response",
                "url": item.get("url"),
                "text": item.get("preview", ""),
            }
        )
        evidence_id += 1

    if not evidence and content:
        evidence.append(
            {
                "evidence_id": evidence_id,
                "type": "content_preview",
                "text": _truncate_text(content, preview_chars),
            }
        )

    content_preview_limit = URL_EVIDENCE_CONTENT_PREVIEW_LIMIT if evidence else preview_chars

    compact = {
        "url": result.get("url"),
        "final_url": result.get("final_url"),
        "title": result.get("title"),
        "task": result.get("task"),
        "domain": result.get("domain"),
        "mode_requested": result.get("mode_requested"),
        "strategy_used": result.get("strategy_used"),
        "confidence": result.get("confidence"),
        "content_chars": result.get("content_chars", 0),
        "content_preview": _truncate_text(content, content_preview_limit),
        "evidence": evidence,
        "found_sections": found_sections,
        "relevant_lines": relevant_lines,
        "table_like_row_count": result.get("table_like_row_count", 0),
        "table_like_rows": table_like_rows,
        "network_response_count": result.get("network_response_count", 0),
        "network_evidence": network_evidence,
        "errors": result.get("errors", []),
        "duration_seconds": result.get("duration_seconds"),
        "retrieval_context": result.get("retrieval_context") or runtime_retrieval_context(),
        "truncated": result.get("truncated", False) or len(content) > preview_chars,
        "answering_instructions": [
            _freshness_instruction(),
            "Answer from the curated evidence, found_sections, relevant_lines, and table_like_rows.",
            "Use network_evidence only when it contains page data, not browser assets.",
            "If evidence is incomplete, say what is missing and what was attempted.",
        ],
    }

    if include_diagnostics:
        compact["diagnostics"] = {
            "labels_used": result.get("labels_used", []),
            "clicked": result.get("clicked", []),
            "scrollable_element_count": result.get("scrollable_element_count", 0),
            "scrollable_elements": result.get("scrollable_elements", [])[:10],
            "strategy_attempts": result.get("strategy_attempts", []),
            "extraction_method": result.get("extraction_method"),
            "playwright_profile": result.get("playwright_profile"),
        }

    if include_raw:
        compact["full_text_preview"] = content
        compact["network_responses"] = result.get("network_responses", [])

    return compact


async def explore_url_pipeline(
    url: str,
    task: str,
    labels: Optional[List[str]] = None,
    mode: str = "auto",
    max_chars: int = DEFAULT_MAX_CHARS,
) -> dict:
    start = time.monotonic()
    max_chars = clamp_int(max_chars, 10000, ABSOLUTE_MAX_CHARS)
    product_bias = is_product_task(task) or _looks_like_product_url(url)
    inferred_labels = infer_page_labels(task=task, headers=labels, product_bias=product_bias)

    attempts = []
    text_parts = []
    errors = []
    title = None
    final_url = url
    clicked = []
    network_responses = []
    scrollable_elements = []
    table_like_rows = []
    best_result = None
    strategy_used = None

    def build_result(profile: str, content: str, playwright_result: Optional[dict] = None) -> dict:
        retrieval_context = runtime_retrieval_context()
        sections = extract_sections_from_text(content, inferred_labels[:50])
        found_sections = {key: value for key, value in sections.items() if value.get("found")}
        rows = table_like_rows or extract_table_like_rows(content, task=task, max_rows=20000)
        relevant_lines = extract_relevant_lines(content, task=task, max_lines=220)

        result = {
            "url": url,
            "final_url": final_url,
            "title": title,
            "task": task,
            "domain": normalize_domain(get_domain(url)),
            "mode_requested": mode,
            "strategy_used": profile,
            "labels_used": inferred_labels,
            "clicked": clicked,
            "scrollable_element_count": len(scrollable_elements),
            "scrollable_elements": scrollable_elements[:50],
            "network_response_count": len(network_responses),
            "network_responses": network_responses,
            "content_chars": len(content),
            "found_sections": found_sections,
            "relevant_lines": relevant_lines,
            "table_like_row_count": len(rows),
            "table_like_rows": rows[:10000],
            "errors": errors,
            "strategy_attempts": attempts,
            "duration_seconds": round(time.monotonic() - start, 2),
            "retrieval_context": retrieval_context,
            "extraction_method": "crawl4ai_direct_playwright_pipeline",
            "full_text_preview": content[:max_chars],
            "truncated": len(content) > max_chars,
        }

        if playwright_result:
            result["playwright_profile"] = playwright_result.get("profile")

        result["confidence"] = estimate_confidence(result)
        result["answering_instructions"] = [
            _freshness_instruction(),
            "Use found_sections first if relevant.",
            "Use table_like_rows for table/list extraction tasks.",
            "Use relevant_lines for concise answer evidence.",
            "Use network response previews for API-sourced data.",
            "If the result is still incomplete, say exactly what is missing and what was attempted.",
        ]

        return result

    try:
        crawl_data = await crawl_url_impl(url)
        crawl_content = extract_content(crawl_data)
        if crawl_content:
            text_parts.append(crawl_content)
        title = extract_title(crawl_data)
        attempts.append(
            {
                "strategy": "crawl4ai_direct",
                "success": bool(crawl_content),
                "content_chars": len(crawl_content),
                "method": crawl_data.get("extraction_method"),
            }
        )
    except Exception as exc:
        errors.append(f"crawl/direct extraction failed: {exc}")
        attempts.append({"strategy": "crawl4ai_direct", "success": False, "error": str(exc)})

    initial_content = "\n\n".join(text_parts)
    if mode == "targeted" and extraction_sufficient(task, build_result("crawl4ai_direct", initial_content)):
        return build_result("crawl4ai_direct", initial_content)

    if mode == "auto":
        profiles = ["targeted", "balanced", "exhaustive"]
    elif mode in {"targeted", "balanced", "exhaustive"}:
        profiles = [mode]
    else:
        profiles = ["targeted", "balanced", "exhaustive"]

    for profile in profiles:
        try:
            dynamic = await playwright_explore_page(url, labels=inferred_labels, task=task, max_chars=max_chars, profile=profile)
            dynamic_content = dynamic.get("content", "")
            if dynamic_content:
                text_parts.append(dynamic_content)

            title = title or dynamic.get("title")
            final_url = dynamic.get("final_url") or final_url
            clicked = unique_preserve_order(clicked + dynamic.get("clicked", []))
            network_responses = dynamic.get("network_responses", [])
            scrollable_elements = dynamic.get("scrollable_elements", [])
            table_like_rows = dynamic.get("table_like_rows", [])
            errors.extend(dynamic.get("errors", []))

            combined = "\n\n".join(part for part in text_parts if part)
            combined = re.sub(r"\n{4,}", "\n\n\n", combined)
            combined = re.sub(r"[ \t]{2,}", " ", combined).strip()

            candidate = build_result(profile, combined, dynamic)
            best_result = candidate
            strategy_used = profile

            attempts.append(
                {
                    "strategy": f"playwright_{profile}",
                    "success": True,
                    "content_chars": dynamic.get("content_chars", 0),
                    "network_response_count": dynamic.get("network_response_count", 0),
                    "scrollable_element_count": dynamic.get("scrollable_element_count", 0),
                    "clicked": dynamic.get("clicked", []),
                    "sufficient": extraction_sufficient(task, candidate),
                }
            )

            if extraction_sufficient(task, candidate):
                return candidate

        except Exception as exc:
            errors.append(f"playwright {profile} extraction failed: {exc}")
            attempts.append({"strategy": f"playwright_{profile}", "success": False, "error": str(exc)})

    combined = "\n\n".join(part for part in text_parts if part)
    combined = re.sub(r"\n{4,}", "\n\n\n", combined)
    combined = re.sub(r"[ \t]{2,}", " ", combined).strip()

    if best_result:
        best_result["strategy_used"] = strategy_used or best_result.get("strategy_used")
        best_result["fallback_exhausted"] = True
        best_result["duration_seconds"] = round(time.monotonic() - start, 2)
        best_result["confidence"] = estimate_confidence(best_result)
        return best_result

    return build_result("failed_all_strategies", combined)


async def crawl_and_ingest(result: dict, query: str, use_browser_fallback: bool = False) -> dict:
    url = result["url"]
    retrieval_context = result.get("retrieval_context") or runtime_retrieval_context()

    content = ""
    title = result.get("title")
    method = None
    errors = []

    try:
        crawl_data = await crawl_url_impl(url)
        content = extract_content(crawl_data)
        title = extract_title(crawl_data, fallback=result.get("title"))
        method = crawl_data.get("extraction_method")
    except Exception as exc:
        errors.append(str(exc))

    if use_browser_fallback and (not content or len(content) < 500):
        try:
            explored = await explore_url_pipeline(url=url, task=query, mode="targeted", max_chars=120000)
            content = explored.get("full_text_preview", "") or content
            title = title or explored.get("title")
            method = explored.get("extraction_method")
        except Exception as exc:
            errors.append(str(exc))

    if not content:
        return {
            "ok": False,
            "url": url,
            "title": result.get("title"),
            "domain": result.get("domain"),
            "retrieval_context": retrieval_context,
            "retrieved_at_utc": retrieval_context.get("retrieved_at_utc"),
            "retrieval_current_date_utc": retrieval_context.get("current_date_utc"),
            "freshness": retrieval_context.get("freshness"),
            "reason": "; ".join(errors) or "No crawlable content returned",
        }

    ingest_result = await rag_ingest_impl(
        IngestRequest(
            text=content,
            metadata={
                "source": url,
                "url": url,
                "title": title,
                "domain": result.get("domain"),
                "query": query,
                "source_score": result.get("score"),
                "source_reason": "; ".join(result.get("score_reasons", [])),
                "content_type": "webpage",
                "retrieved_at_utc": retrieval_context.get("retrieved_at_utc"),
                "retrieval_current_date_utc": retrieval_context.get("current_date_utc"),
            },
        )
    )

    return {
        "ok": True,
        "title": title,
        "url": url,
        "domain": result.get("domain"),
        "stored_chunks": ingest_result.get("stored", 0),
        "content_chars": len(content),
        "source_score": result.get("score"),
        "source_reason": result.get("score_reasons", []),
        "extraction_method": method,
        "retrieval_context": retrieval_context,
        "retrieved_at_utc": retrieval_context.get("retrieved_at_utc"),
        "retrieval_current_date_utc": retrieval_context.get("current_date_utc"),
        "freshness": retrieval_context.get("freshness"),
    }


def build_evidence_pack(results: List[dict]) -> List[dict]:
    evidence = []

    for index, item in enumerate(results, start=1):
        text = item.get("text") or ""

        evidence.append(
            {
                "evidence_id": index,
                "title": item.get("title"),
                "url": item.get("url") or item.get("source"),
                "domain": item.get("domain"),
                "section": item.get("section"),
                "quote": text[:1600],
                "vector_score": item.get("vector_score"),
                "rerank_score": item.get("rerank_score"),
                "ingested_at": item.get("ingested_at"),
                "retrieved_at_utc": item.get("retrieved_at_utc") or item.get("ingested_at"),
            }
        )

    return evidence


async def research_pipeline(
    query: str,
    mode: str = "balanced",
    max_sources: Optional[int] = None,
    verify: bool = True,
) -> dict:
    start = time.monotonic()
    retrieval_context = runtime_retrieval_context()
    mode = mode if mode in RESEARCH_MODE_CONFIG else "balanced"
    config = RESEARCH_MODE_CONFIG[mode]

    # Keep each mode inside its intended latency envelope. MCP/SSE clients often
    # close long-running tool calls, so a balanced request should not become a
    # 10-source crawl just because the caller provided a high max_sources value.
    max_urls_value = config["max_urls"] if max_sources is None else clamp_int(max_sources, 0, config["max_urls"])
    search_results_value = config["search_results"]
    top_k_value = config["top_k"]

    if mode == "local_only":
        rag_result = await rag_query_impl(QueryRequest(query=query, top_k=top_k_value))
        return {
            "query": query,
            "mode": mode,
            "retrieval_context": retrieval_context,
            "searched": [],
            "selected_for_crawl": [],
            "crawled_sources": [],
            "failed_sources": [],
            "evidence": build_evidence_pack(rag_result.get("results", [])),
            "results": rag_result.get("results", []),
            "answering_instructions": [
                _freshness_instruction(),
                "Answer from the returned local memory evidence.",
                "Cite source URLs where available.",
                "If memory does not contain enough evidence, say that web research may be needed.",
            ],
            "duration_seconds": round(time.monotonic() - start, 2),
        }

    try:
        candidates = await searxng_search(query=query, max_results=search_results_value, mode=mode)
        candidates = _stamp_retrieval_context(candidates, retrieval_context)
        search_error = None
    except Exception as exc:
        candidates = []
        search_error = str(exc)
        logger.error("SearXNG search failed in research: %s", search_error)

    selected = candidates[:max_urls_value]

    crawled_sources = []
    failed_sources = []

    if selected:
        use_browser_fallback = mode in {"deep", "technical", "academic"} or verify
        tasks = [crawl_and_ingest(result, query=query, use_browser_fallback=use_browser_fallback) for result in selected]
        crawl_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result, original in zip(crawl_results, selected):
            if isinstance(result, Exception):
                failed_sources.append(
                    {
                        "url": original["url"],
                        "title": original["title"],
                        "domain": original["domain"],
                        "retrieval_context": original.get("retrieval_context") or retrieval_context,
                        "retrieved_at_utc": original.get("retrieved_at_utc") or retrieval_context.get("retrieved_at_utc"),
                        "retrieval_current_date_utc": original.get("retrieval_current_date_utc") or retrieval_context.get("current_date_utc"),
                        "freshness": original.get("freshness") or retrieval_context.get("freshness"),
                        "reason": str(result),
                    }
                )
            elif result.get("ok"):
                result.pop("ok", None)
                crawled_sources.append(result)
            else:
                result.pop("ok", None)
                failed_sources.append(result)

    rag_results = []
    if top_k_value > 0:
        try:
            rag_result = await rag_query_impl(QueryRequest(query=query, top_k=top_k_value))
            rag_results = rag_result.get("results", [])
        except Exception as exc:
            logger.error("RAG query failed in research: %s", str(exc))

    response = {
        "query": query,
        "mode": mode,
        "retrieval_context": retrieval_context,
        "searched": candidates,
        "selected_for_crawl": selected,
        "crawled_sources": crawled_sources,
        "failed_sources": failed_sources,
        "evidence": build_evidence_pack(rag_results),
        "results": rag_results,
        "answering_instructions": [
            _freshness_instruction(),
            "Use evidence for factual claims.",
            "Cite URLs inline.",
            "Mention uncertainty if sources conflict or evidence is incomplete.",
            "Prefer official, primary, technical, or authoritative sources.",
        ],
        "duration_seconds": round(time.monotonic() - start, 2),
    }

    if search_error:
        response["search_error"] = search_error

    return response
