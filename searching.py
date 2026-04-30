import re
from typing import Any, Dict, List

import httpx

from shared import SEARXNG_URL, get_domain

RESEARCH_MODE_CONFIG = {
    "quick": {"max_urls": 2, "search_results": 6, "top_k": 4},
    "balanced": {"max_urls": 4, "search_results": 10, "top_k": 6},
    "deep": {"max_urls": 8, "search_results": 16, "top_k": 10},
    "technical": {"max_urls": 6, "search_results": 14, "top_k": 8},
    "academic": {"max_urls": 6, "search_results": 14, "top_k": 8},
    "local_only": {"max_urls": 0, "search_results": 0, "top_k": 8},
    "web_only": {"max_urls": 5, "search_results": 12, "top_k": 0},
}

DOMAIN_BOOSTS = {
    "github.com": 3.0,
    "docs.python.org": 3.0,
    "developer.mozilla.org": 3.0,
    "kubernetes.io": 3.0,
    "docs.docker.com": 3.0,
    "docs.github.com": 3.0,
    "learn.microsoft.com": 2.5,
    "cloud.google.com": 2.2,
    "docs.aws.amazon.com": 2.2,
    "stackoverflow.com": 2.0,
    "serverfault.com": 2.0,
    "superuser.com": 1.8,
    "unix.stackexchange.com": 2.0,
    "askubuntu.com": 1.8,
    "wiki.archlinux.org": 2.5,
    "man7.org": 2.3,
    "mankier.com": 2.0,
    "arxiv.org": 2.2,
    "semanticscholar.org": 2.0,
    "pubmed.ncbi.nlm.nih.gov": 2.0,
    "wikipedia.org": 0.7,
    "fleetguard.com": 2.5,
    "cummins.com": 2.5,
}

DOMAIN_PENALTIES = {
    "pinterest.com": -5.0,
    "quora.com": -2.0,
    "medium.com": -0.8,
    "dev.to": -0.3,
    "fandom.com": -4.0,
    "fiction.live": -5.0,
    "archiveofourown.org": -5.0,
    "reddit.com": -0.5,
    "x.com": -2.0,
    "twitter.com": -2.0,
    "facebook.com": -4.0,
    "instagram.com": -4.0,
    "tiktok.com": -4.0,
}

BLOCKED_DOMAINS = {
    "pinterest.com",
    "fandom.com",
    "fiction.live",
    "archiveofourown.org",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
}

TECHNICAL_DOMAINS = {
    "github.com",
    "stackoverflow.com",
    "serverfault.com",
    "superuser.com",
    "unix.stackexchange.com",
    "askubuntu.com",
    "wiki.archlinux.org",
    "docs.docker.com",
    "kubernetes.io",
    "docs.python.org",
    "developer.mozilla.org",
    "learn.microsoft.com",
    "man7.org",
    "mankier.com",
}

ACADEMIC_DOMAINS = {
    "arxiv.org",
    "semanticscholar.org",
    "pubmed.ncbi.nlm.nih.gov",
    "doi.org",
    "crossref.org",
}


def normalize_domain(domain: str) -> str:
    domain = (domain or "").lower().strip()
    return domain[4:] if domain.startswith("www.") else domain


def strip_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def score_search_result(result: Dict[str, Any], query: str, mode: str = "balanced") -> Dict[str, Any]:
    title = result.get("title") or ""
    url = result.get("url") or ""
    snippet = result.get("content") or result.get("snippet") or ""
    engine = result.get("engine")
    domain = normalize_domain(get_domain(url))

    score = 1.0
    reasons = []

    if domain in DOMAIN_BOOSTS:
        score += DOMAIN_BOOSTS[domain]
        reasons.append(f"domain boost: {domain}")

    if domain in DOMAIN_PENALTIES:
        score += DOMAIN_PENALTIES[domain]
        reasons.append(f"domain penalty: {domain}")

    if mode == "technical" and domain in TECHNICAL_DOMAINS:
        score += 2.0
        reasons.append("technical source")

    if mode == "academic" and domain in ACADEMIC_DOMAINS:
        score += 2.0
        reasons.append("academic source")

    lowered = f"{title} {snippet} {url}".lower()
    query_terms = [term.lower() for term in re.findall(r"[a-zA-Z0-9_\-\.]{3,}", query)]

    if query_terms:
        matches = sum(1 for term in query_terms if term in lowered)
        score += min(matches * 0.25, 2.0)
        if matches:
            reasons.append(f"query term matches: {matches}")

    if "/wiki/portal:current_events" in url.lower():
        score -= 4.0
        reasons.append("current-events portal penalty")

    if "sandbox" in lowered or "alternate history" in lowered or "fiction" in lowered:
        score -= 3.0
        reasons.append("fiction/sandbox penalty")

    if not snippet:
        score -= 0.5
        reasons.append("missing snippet penalty")

    if engine in {"github", "stackoverflow", "arxiv"}:
        score += 0.8
        reasons.append(f"engine boost: {engine}")

    result["score"] = round(score, 3)
    result["score_reasons"] = reasons
    return result


def compact_search_results(data: dict, query: str, max_results: int = 10, mode: str = "balanced") -> List[dict]:
    seen_urls = set()
    results = []

    for item in data.get("results", []):
        url = item.get("url")
        title = item.get("title")
        content = item.get("content") or ""

        if not url or not title:
            continue

        url = url.strip()
        if url in seen_urls:
            continue

        domain = normalize_domain(get_domain(url))
        if domain in BLOCKED_DOMAINS:
            continue

        seen_urls.add(url)

        result = {
            "title": strip_text(title),
            "url": url,
            "domain": domain,
            "snippet": strip_text(content)[:900],
            "engine": item.get("engine"),
        }

        results.append(score_search_result(result, query=query, mode=mode))

    results.sort(key=lambda item: item.get("score", 0), reverse=True)
    return results[:max_results]


async def searxng_search(query: str, max_results: int = 10, mode: str = "balanced") -> List[dict]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{SEARXNG_URL}/search", params={"q": query, "format": "json"})
        resp.raise_for_status()
        data = resp.json()

    return compact_search_results(data, query=query, max_results=max_results, mode=mode)
