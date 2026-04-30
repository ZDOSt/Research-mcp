import asyncio
import html
import json
import re
from typing import List, Optional
from urllib.parse import urlparse

from playwright.async_api import async_playwright

from crawler import BROWSER_HEADERS
from extractors import extract_table_like_rows, html_to_text, parse_maybe_json_text, unique_preserve_order

DEFAULT_MAX_CHARS = 300000
ABSOLUTE_MAX_CHARS = 750000
NETWORK_BODY_LIMIT = 1_000_000
NETWORK_TEXT_LIMIT = 8_000
NETWORK_PREVIEW_LIMIT = 600
NETWORK_COMBINED_TEXT_LIMIT = 4_000
MAX_NETWORK_RESPONSES = 6
MAX_NETWORK_CANDIDATES = 40
NETWORK_MIN_SCORE = 8

STATIC_URL_RE = re.compile(
    r"\.(?:js|mjs|css|png|jpe?g|gif|webp|svg|ico|woff2?|ttf|otf|mp4|webm|m3u8|ts|map)(?:[?#]|$)",
    re.I,
)
HARD_NOISY_URL_MARKERS = [
    "adtech", "advertis", "analytics", "beacon", "brightline.tv", "cookielaw",
    "comscore", "consent", "doubleclick", "fave", "googletag", "gtm.js",
    "hotjar", "onetrust", "optimizely", "prebid", "scorecardresearch",
    "segment.io", "sentry", "sourcepoint", "tinypass", "tracking",
    "web-vitals", "widgetapi", "youtube.com/iframe",
]
SOFT_NOISY_URL_MARKERS = [
    "/assets/", "/bundles/", "/dist/", "/static/", "bootstrap", "chunk-",
    "feature-flag", "font", "metrics", "player", "polyfill", "runtime-config",
    "session-context", "telemetry", "token", "vendor", "webpack",
]
STATIC_RESOURCE_TYPES = {"script", "stylesheet", "image", "media", "font", "manifest"}
CAPTURABLE_RESOURCE_TYPES = {"xhr", "fetch"}
STATIC_CONTENT_TYPE_MARKERS = [
    "javascript", "ecmascript", "text/css", "font/", "image/", "video/", "audio/",
    "mpegurl", "dash+xml", "octet-stream",
]
DATA_PATH_SEGMENTS = {
    "api", "graphql", "gql", "content", "contents", "article", "articles",
    "story", "stories", "live", "search", "product", "products", "commerce",
    "connect", "aura", "apexremote", "sfsites", "webruntime", "services",
    "catalog", "reference", "references", "spec", "specs", "specification",
    "specifications", "equipment", "maintenance", "docs", "documentation",
    "download", "downloads", "extension", "extensions", "plugin", "plugins",
    "manifest", "package", "packages", "readme", "repo", "repos", "repository",
    "raw", "registry", "release", "releases", "changelog", "schema", "schemas",
    "config", "configs", "settings", "source", "sources", "metadata", "version",
    "versions", "module", "modules", "compose", "list", "lists", "table",
    "tables", "data", "dataset", "datasets", "feed", "feeds",
}
DATA_FILE_RE = re.compile(r"\.(?:json|ndjson|graphql|xml)(?:[?#]|$)", re.I)
NETWORK_RELEVANCE_STOP_WORDS = {
    "about", "after", "also", "answer", "before", "compare", "details", "events",
    "extract", "find", "from", "guide", "help", "information", "into", "learn",
    "made", "more", "news", "overview", "page", "please", "random", "reference",
    "search", "show", "summarize", "that", "this", "using", "verify", "what",
    "when", "where", "with",
}

_browser_semaphore = asyncio.Semaphore(1)


def clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def normalize_heading(text: str) -> str:
    text = html.unescape(text or "")
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def root_domain(domain: str) -> str:
    parts = (domain or "").lower().split(".")
    if len(parts) <= 2:
        return ".".join(parts)
    return ".".join(parts[-2:])


def response_path_query(response_url: str) -> str:
    parsed = urlparse(response_url)
    return f"{parsed.path}?{parsed.query}".lower()


def response_path_tokens(response_url: str) -> set:
    return set(re.findall(r"[a-z0-9]{2,}", response_path_query(response_url)))


def has_noisy_network_signal(response_url: str) -> bool:
    lower_url = response_url.lower()
    return bool(STATIC_URL_RE.search(lower_url)) or any(marker in lower_url for marker in HARD_NOISY_URL_MARKERS)


def has_soft_noisy_network_signal(response_url: str) -> bool:
    lower_url = response_url.lower()
    return any(marker in lower_url for marker in SOFT_NOISY_URL_MARKERS)


def has_data_endpoint_signal(response_url: str) -> bool:
    path_query = response_path_query(response_url)
    if DATA_FILE_RE.search(path_query):
        return True

    tokens = response_path_tokens(response_url)
    return bool(tokens & DATA_PATH_SEGMENTS)


def network_relevance_terms(task: Optional[str], labels: Optional[List[str]]) -> List[str]:
    label_text = " ".join(labels or []) if labels and len(labels) <= 12 else ""
    text = " ".join([task or "", label_text])
    terms = []

    for term in re.findall(r"[a-z0-9][a-z0-9-]{2,}", text.lower()):
        if len(term) < 4 or term in NETWORK_RELEVANCE_STOP_WORDS:
            continue
        terms.append(term)

    return unique_preserve_order(terms)[:25]


def should_capture_network_response(
    response_url: str,
    content_type: str,
    resource_type: str,
    start_domain: str,
) -> bool:
    lower_url = response_url.lower()
    content_type = (content_type or "").lower()
    resource_type = (resource_type or "").lower()

    if resource_type in STATIC_RESOURCE_TYPES:
        return False
    if has_noisy_network_signal(lower_url):
        return False
    if any(marker in content_type for marker in STATIC_CONTENT_TYPE_MARKERS):
        return False
    if resource_type and resource_type not in CAPTURABLE_RESOURCE_TYPES:
        return False

    is_json = "json" in content_type or "graphql" in content_type
    is_xml = "xml" in content_type
    is_text = "text/plain" in content_type
    if not (is_json or is_xml or is_text):
        return False

    # DOM extraction already captures rendered HTML. Treat network capture as a
    # data-channel only, otherwise browser/framework assets become "evidence".
    if "text/html" in content_type:
        return False

    parsed = urlparse(response_url)
    response_domain = parsed.netloc.lower()
    same_site = root_domain(response_domain) == root_domain(start_domain)
    data_endpoint = has_data_endpoint_signal(response_url)

    if same_site:
        return data_endpoint or is_json or is_xml

    return data_endpoint and (is_json or is_xml)


def looks_like_script_or_config(text: str) -> bool:
    sample = (text or "").lstrip()[:4000]
    lower = sample.lower()
    script_markers = [
        "webpack", "function(", "=>", "window.", "document.", "createscript",
        "sourcemappingurl", "__webpack_require__", "define(", "var ", "const ",
    ]
    if any(marker in lower for marker in script_markers) and sample.count(";") >= 8:
        return True

    return False


def score_network_response(
    item: dict,
    start_domain: str,
    task: Optional[str],
    labels: Optional[List[str]],
) -> int:
    response_url = item.get("url") or ""
    content_type = (item.get("content_type") or "").lower()
    parsed = urlparse(response_url)
    response_domain = parsed.netloc.lower()
    same_site = root_domain(response_domain) == root_domain(start_domain)
    data_endpoint = has_data_endpoint_signal(response_url)
    text = item.get("text") or item.get("preview") or ""
    lower_text = text.lower()
    lower_url = response_url.lower()
    status = item.get("status") or 0

    score = 0
    if 200 <= int(status) < 300:
        score += 1
    if "json" in content_type or "graphql" in content_type:
        score += 4
    elif "xml" in content_type:
        score += 3
    elif "text/plain" in content_type:
        score += 1
    if same_site:
        score += 2
    if data_endpoint:
        score += 3

    term_hits = 0
    for term in network_relevance_terms(task, labels):
        if term in lower_text or term in lower_url:
            term_hits += 1
    score += min(term_hits, 5) * 2

    if len(text.strip()) < 80 and term_hits == 0:
        score -= 2
    if has_noisy_network_signal(response_url) or looks_like_script_or_config(text):
        score -= 20
    elif has_soft_noisy_network_signal(response_url) and term_hits == 0:
        score -= 8

    item["_network_score"] = score
    item["_network_term_hits"] = term_hits
    item["_network_data_endpoint"] = data_endpoint
    item["_network_same_site"] = same_site
    return score


def select_network_responses(
    responses: List[dict],
    start_domain: str,
    task: Optional[str],
    labels: Optional[List[str]],
) -> List[dict]:
    deduped = []
    seen_urls = set()

    for item in responses:
        response_url = item.get("url")
        if not response_url or response_url in seen_urls:
            continue
        seen_urls.add(response_url)

        score = score_network_response(item, start_domain, task, labels)
        if score < NETWORK_MIN_SCORE:
            continue
        if not item.get("_network_same_site") and not item.get("_network_term_hits"):
            continue
        if not item.get("_network_data_endpoint") and not item.get("_network_term_hits"):
            continue

        deduped.append(item)

    deduped.sort(key=lambda item: item.get("_network_score", 0), reverse=True)
    return deduped[:MAX_NETWORK_RESPONSES]



def build_click_script(labels: List[str]) -> str:
    labels_json = json.dumps([normalize_heading(label) for label in labels])

    return f"""
    async () => {{
      const labels = {labels_json};
      const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
      const normalize = (text) => String(text || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').replace(/\\s+/g, ' ').trim();
      const visibleText = (el) => (
        el.innerText ||
        el.textContent ||
        el.getAttribute('aria-label') ||
        el.getAttribute('title') ||
        el.getAttribute('data-label') ||
        el.getAttribute('name') ||
        ''
      ).trim();

      const selector = [
        'button',
        'a',
        '[role="tab"]',
        '[role="button"]',
        'summary',
        '[aria-controls]',
        '[data-toggle]',
        '[data-bs-toggle]',
        '[data-testid]',
        '[data-tab]',
        '.tab',
        '.tabs',
        '.accordion',
        '.accordion-button',
        'nav *',
        '[class*="tab"]',
        '[class*="accordion"]',
        '[class*="load"]',
        '[class*="more"]'
      ].join(',');

      const candidates = Array.from(document.querySelectorAll(selector));
      const clicked = [];

      for (const label of labels) {{
        for (const el of candidates) {{
          const text = normalize(visibleText(el));
          if (!text) continue;
          if (clicked.includes(text)) continue;

          const matches = text.includes(label) || label.includes(text);

          if (matches) {{
            try {{
              el.scrollIntoView({{block: 'center', inline: 'center'}});
              await sleep(500);
              el.click();
              clicked.push(text);
              await sleep(1600);
            }} catch (e) {{}}
          }}
        }}
      }}

      return {{
        clicked,
        title: document.title,
        url: location.href,
        text: document.body ? document.body.innerText : ''
      }};
    }}
    """


def build_scrollable_capture_script(labels: List[str], profile: str) -> str:
    labels_json = json.dumps([normalize_heading(label) for label in labels])
    page_steps = 8 if profile == "targeted" else 14 if profile == "balanced" else 24
    container_steps = 8 if profile == "targeted" else 14 if profile == "balanced" else 24
    max_elements = 15 if profile == "targeted" else 30 if profile == "balanced" else 60

    return f"""
    async () => {{
      const labels = {labels_json};
      const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
      const normalize = (text) => String(text || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').replace(/\\s+/g, ' ').trim();

      const isScrollable = (el) => {{
        if (!el) return false;
        const style = window.getComputedStyle(el);
        const overflowY = style.overflowY;
        const overflowX = style.overflowX;
        const canScrollY = el.scrollHeight > el.clientHeight + 20 && ['auto', 'scroll', 'overlay'].includes(overflowY);
        const canScrollX = el.scrollWidth > el.clientWidth + 20 && ['auto', 'scroll', 'overlay'].includes(overflowX);
        return canScrollY || canScrollX;
      }};

      const textNearLabels = (el) => {{
        const text = normalize(el.innerText || el.textContent || '');
        if (!labels.length) return true;
        return labels.some(label => text.includes(label));
      }};

      const all = Array.from(document.querySelectorAll('body, main, section, article, div, table, tbody, [role="table"], [role="grid"], [class*="table"], [class*="grid"], [class*="scroll"], [class*="list"], [class*="result"], [class*="data"]'));
      const scrollables = all.filter(isScrollable).slice(0, {max_elements});
      const snapshots = [];
      const used = [];

      for (const el of scrollables) {{
        try {{
          const text = normalize(el.innerText || el.textContent || '');
          const relevant = textNearLabels(el) || text.length > 200;
          if (!relevant) continue;

          used.push({{
            tag: el.tagName,
            className: el.className ? String(el.className).slice(0, 200) : '',
            id: el.id || '',
            scrollHeight: el.scrollHeight,
            clientHeight: el.clientHeight,
            scrollWidth: el.scrollWidth,
            clientWidth: el.clientWidth
          }});

          const maxY = Math.max(0, el.scrollHeight - el.clientHeight);
          const maxX = Math.max(0, el.scrollWidth - el.clientWidth);
          const ySteps = {container_steps};
          const xSteps = maxX > 100 ? 4 : 1;

          for (let yi = 0; yi <= ySteps; yi++) {{
            el.scrollTop = Math.floor(maxY * yi / ySteps);

            for (let xi = 0; xi <= xSteps; xi++) {{
              el.scrollLeft = Math.floor(maxX * xi / Math.max(1, xSteps));
              await sleep(300);
              const part = el.innerText || el.textContent || '';
              if (part && part.trim()) snapshots.push(part);
            }}
          }}
        }} catch (e) {{}}
      }}

      const pageSnapshots = [];
      const pageSteps = {page_steps};
      for (let i = 0; i <= pageSteps; i++) {{
        window.scrollTo(0, Math.floor(document.body.scrollHeight * i / pageSteps));
        await sleep(400);
        pageSnapshots.push(document.body ? document.body.innerText : '');
      }}

      window.scrollTo(0, 0);
      await sleep(500);

      return {{
        title: document.title,
        url: location.href,
        scrollable_elements: used,
        text: [...pageSnapshots, ...snapshots].join('\\n\\n')
      }};
    }}
    """


async def playwright_explore_page(
    url: str,
    labels: Optional[List[str]] = None,
    task: Optional[str] = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    profile: str = "targeted",
    timeout_ms: int = 60000,
) -> dict:
    async with _browser_semaphore:
        return await _playwright_explore_page_inner(
            url=url,
            labels=labels,
            task=task,
            max_chars=max_chars,
            profile=profile,
            timeout_ms=timeout_ms,
        )


async def _playwright_explore_page_inner(
    url: str,
    labels: Optional[List[str]] = None,
    task: Optional[str] = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    profile: str = "targeted",
    timeout_ms: int = 60000,
) -> dict:
    labels = labels or []
    max_chars = clamp_int(max_chars, 10000, ABSOLUTE_MAX_CHARS)
    profile = profile if profile in {"targeted", "balanced", "exhaustive"} else "targeted"

    captured_responses = []
    capture_tasks = set()
    clicked = []
    errors = []
    dom_text_parts = []
    scrollable_elements = []
    title = None
    final_url = url

    parsed_start = urlparse(url)
    start_domain = parsed_start.netloc.lower()

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu", "--disable-web-security"],
        )

        context = await browser.new_context(
            user_agent=BROWSER_HEADERS["User-Agent"],
            locale="en-US",
            viewport={"width": 1600, "height": 1400},
            ignore_https_errors=True,
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )

        page = await context.new_page()

        async def capture_response(response):
            try:
                if len(captured_responses) >= MAX_NETWORK_CANDIDATES:
                    return

                response_url = response.url
                content_type = (response.headers.get("content-type") or "").lower()
                resource_type = getattr(response.request, "resource_type", "")

                if not should_capture_network_response(response_url, content_type, resource_type, start_domain):
                    return

                body = await response.text()
                if not body or len(body) > NETWORK_BODY_LIMIT:
                    return

                text = parse_maybe_json_text(body) if "json" in content_type or body.strip().startswith(("{", "[")) else html_to_text(body)

                if text:
                    if len(captured_responses) >= MAX_NETWORK_CANDIDATES:
                        return
                    if looks_like_script_or_config(text):
                        return
                    captured_responses.append(
                        {
                            "url": response_url,
                            "status": response.status,
                            "content_type": content_type,
                            "resource_type": resource_type,
                            "text": text[:NETWORK_TEXT_LIMIT],
                            "text_chars": len(text),
                        }
                    )

            except Exception:
                return

        def schedule_capture(response):
            task = asyncio.create_task(capture_response(response))
            capture_tasks.add(task)
            task.add_done_callback(capture_tasks.discard)

        page.on("response", schedule_capture)

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                await page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass
            await page.wait_for_timeout(2500)

            title = await page.title()
            final_url = page.url

            try:
                initial_text = await page.locator("body").inner_text(timeout=15000)
                if initial_text:
                    dom_text_parts.append(initial_text)
            except Exception:
                pass

            page_steps = 6 if profile == "targeted" else 12 if profile == "balanced" else 24
            for i in range(0, page_steps + 1):
                pct = i / page_steps
                await page.evaluate(f"window.scrollTo(0, Math.floor(document.body.scrollHeight * {pct}))")
                await page.wait_for_timeout(450)

                try:
                    text = await page.locator("body").inner_text(timeout=8000)
                    if text:
                        dom_text_parts.append(text)
                except Exception:
                    pass

            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(800)

            if labels:
                result = await page.evaluate(build_click_script(labels))
                if isinstance(result, dict):
                    clicked.extend(result.get("clicked") or [])
                    if result.get("title"):
                        title = result["title"]
                    if result.get("url"):
                        final_url = result["url"]
                    if result.get("text"):
                        dom_text_parts.append(result["text"])

                await page.wait_for_timeout(3500 if profile == "targeted" else 5000)

            if profile in {"balanced", "exhaustive"}:
                generic_labels = [
                    "show more", "view more", "load more", "more", "details", "learn more",
                    "expand",
                ]

                result = await page.evaluate(build_click_script(generic_labels))
                if isinstance(result, dict):
                    clicked.extend(result.get("clicked") or [])
                    if result.get("text"):
                        dom_text_parts.append(result["text"])

                await page.wait_for_timeout(3000)

            scroll_result = await page.evaluate(build_scrollable_capture_script(labels, profile))
            if isinstance(scroll_result, dict):
                if scroll_result.get("title"):
                    title = scroll_result["title"]
                if scroll_result.get("url"):
                    final_url = scroll_result["url"]
                if scroll_result.get("text"):
                    dom_text_parts.append(scroll_result["text"])
                if scroll_result.get("scrollable_elements"):
                    scrollable_elements.extend(scroll_result["scrollable_elements"])

            await page.wait_for_timeout(2000)

            try:
                final_text = await page.locator("body").inner_text(timeout=12000)
                if final_text:
                    dom_text_parts.append(final_text)
            except Exception:
                pass

            if capture_tasks:
                done, pending = await asyncio.wait(capture_tasks, timeout=5)
                for task in pending:
                    task.cancel()

        except Exception as exc:
            errors.append(str(exc))

        await context.close()
        await browser.close()

    captured_responses = select_network_responses(captured_responses, start_domain, task, labels)
    network_text = "\n\n".join(
        f"Network response: {item['url']}\n{item['text'][:NETWORK_COMBINED_TEXT_LIMIT]}"
        for item in captured_responses
        if item.get("text")
    )
    dom_text = "\n\n".join(dom_text_parts)
    combined = "\n\n".join(part for part in [dom_text, network_text] if part)
    combined = html.unescape(combined)
    combined = re.sub(r"\n{4,}", "\n\n\n", combined)
    combined = re.sub(r"[ \t]{2,}", " ", combined).strip()

    table_like_rows = extract_table_like_rows(combined, max_rows=20000)

    return {
        "url": url,
        "final_url": final_url,
        "title": title,
        "profile": profile,
        "clicked": unique_preserve_order(clicked),
        "scrollable_elements": scrollable_elements[:50],
        "scrollable_element_count": len(scrollable_elements),
        "content": combined[:max_chars],
        "content_chars": len(combined),
        "truncated": len(combined) > max_chars,
        "table_like_rows": table_like_rows[:10000],
        "table_like_row_count": len(table_like_rows),
        "network_responses": [
            {
                "url": item["url"],
                "status": item["status"],
                "content_type": item["content_type"],
                "resource_type": item.get("resource_type"),
                "text_chars": item["text_chars"],
                "preview": item["text"][:NETWORK_PREVIEW_LIMIT],
            }
            for item in captured_responses[:MAX_NETWORK_RESPONSES]
        ],
        "network_response_count": len(captured_responses),
        "errors": errors,
        "extraction_method": f"playwright_{profile}_browser_network_capture_scrollable_tables",
    }
