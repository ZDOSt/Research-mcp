import html
import json
import re
from typing import Any, Dict, List, Optional

GENERAL_SECTION_LABELS = [
    "Overview", "Documentation", "Docs", "Guide", "Guides", "Getting Started", "Quickstart",
    "Installation", "Install", "Setup", "Configuration", "Configure", "Usage", "Examples",
    "API", "API Reference", "Reference", "Tutorial", "How To", "Troubleshooting", "FAQ",
    "Requirements", "Compatibility", "Security", "Authentication", "Authorization",
    "Integrations", "Pricing", "Plans", "Downloads", "Resources", "Documents",
    "Release Notes", "Releases", "Changelog", "Versions", "Migration", "Upgrade",
    "Limitations", "Known Issues", "Support",
]

BROAD_DOCUMENTATION_LABELS = [
    "Overview", "Documentation", "Docs", "Guide", "Guides", "Getting Started", "Quickstart",
    "Installation", "Install", "Setup", "Configuration", "Configure", "Usage", "Examples",
    "Reference", "Tutorial", "How To", "Troubleshooting", "FAQ", "Requirements",
    "Compatibility", "Integrations", "Downloads", "Resources", "Documents",
    "Release Notes", "Releases", "Changelog", "Versions", "Migration", "Upgrade",
    "Limitations", "Known Issues", "Support",
]

PRODUCT_SECTION_LABELS = [
    "Specifications", "Specification", "Specs", "Technical Specifications",
    "Product Specifications", "Attributes", "Product Details", "Equipment", "Applications",
    "Application", "Equipment Applications", "Fits", "Used On", "Compatibility",
    "OEM Cross Reference", "Cross Reference", "Cross References", "Interchange",
    "Interchanges", "Replacement", "Replaces", "Equivalent", "Equivalent Parts",
    "Competitor Cross Reference", "Part Cross Reference", "OE Cross Reference",
    "Maintenance Kits", "Maintenance Kit", "Kits", "Service Kits", "Repair Kits", "Related Kits",
]

REVEAL_CONTROL_LABELS = ["Show more", "View more", "Load more", "More", "Details", "Learn more", "Expand"]

SECTION_INTENT_ALIASES = {
    "install": ["Install", "Installation", "Setup", "Getting Started", "Quickstart"],
    "configure": ["Configuration", "Configure", "Settings", "Options", "Environment Variables"],
    "usage": ["Usage", "Examples", "Guide", "Tutorial", "How To"],
    "api": ["API", "API Reference", "Reference", "Endpoints", "Authentication", "Authorization"],
    "troubleshooting": ["Troubleshooting", "FAQ", "Known Issues", "Limitations", "Support"],
    "download": ["Downloads", "Resources", "Documents", "Files", "PDF", "Manuals"],
    "release": ["Release Notes", "Releases", "Changelog", "Versions", "Migration", "Upgrade"],
    "pricing": ["Pricing", "Plans", "Billing"],
    "security": ["Security", "Authentication", "Authorization", "Permissions"],
    "compatibility": ["Compatibility", "Requirements", "Supported Platforms", "Versions"],
    "compose": [
        "Docker Compose", "Compose", "Compose file", "Compose file reference",
        "Services", "Volumes", "Networks", "Environment Variables",
    ],
    "extension": [
        "Extensions", "Extension", "Plugins", "Plugin", "Add-ons", "Addons",
        "Development", "Developer Guide", "Manifest", "Examples",
    ],
    "schema": ["Schema", "Schemas", "Manifest", "Configuration", "Reference", "Options", "Settings"],
    "specifications": [
        "Specifications", "Specification", "Specs", "Technical Specifications",
        "Product Specifications", "Attributes", "Product Details",
    ],
    "equipment": ["Equipment", "Applications", "Application", "Equipment Applications", "Fits", "Used On", "Compatibility"],
    "cross_reference": [
        "OEM Cross Reference", "Cross Reference", "Cross References", "Interchange",
        "Interchanges", "Replacement", "Replaces", "Equivalent", "Equivalent Parts",
        "Competitor Cross Reference", "Part Cross Reference", "OE Cross Reference",
    ],
    "maintenance_kits": ["Maintenance Kits", "Maintenance Kit", "Kits", "Service Kits", "Repair Kits", "Related Kits"],
}

COMMON_STOP_HEADERS = {label.lower() for label in GENERAL_SECTION_LABELS + PRODUCT_SECTION_LABELS}
COMMON_STOP_HEADERS.update({
    "description", "features", "details", "parts", "related parts", "related products",
    "images", "reviews", "where to buy",
})


def clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def normalize_heading(text: str) -> str:
    text = html.unescape(text or "")
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    output = []

    for item in items:
        item = str(item).strip()
        if not item:
            continue

        key = item.lower()
        if key in seen:
            continue

        seen.add(key)
        output.append(item)

    return output


def is_product_task(task: Optional[str]) -> bool:
    text = (task or "").lower()
    product_terms = [
        "product", "part", "part number", "sku", "oem", "oe", "cross reference", "xref",
        "interchange", "replacement", "replaces", "equivalent", "specification",
        "specifications", "equipment", "application", "maintenance kit", "service kit", "filter",
    ]
    return any(term in text for term in product_terms)


def is_documentation_task(task: Optional[str]) -> bool:
    text = (task or "").lower()
    documentation_terms = [
        "api", "auth", "authentication", "authorization", "changelog", "config",
        "configuration", "configure", "container", "compose", "create", "deploy",
        "deployment", "develop", "docker", "docs", "documentation", "download",
        "endpoint", "environment", "example", "extension", "faq", "github",
        "guide", "implement", "install", "installation", "integration", "manual",
        "manifest", "migration", "package", "plugin", "quickstart", "readme",
        "reference", "release notes", "repository", "requirements", "schema",
        "setup", "template", "troubleshoot", "tutorial", "upgrade", "usage",
        "version", "yaml", "yml",
    ]
    documentation_phrases = [
        "best practice", "best practices", "how can", "how do", "how should",
        "how to", "show me how", "compose file", "docker compose", "self host",
        "self-host",
    ]
    return any(term in text for term in documentation_terms) or any(phrase in text for phrase in documentation_phrases)


def infer_page_labels(task: Optional[str] = None, headers: Optional[List[str]] = None, product_bias: bool = False) -> List[str]:
    labels = []

    if headers:
        labels.extend(headers)

    task_text = (task or "").lower()
    documentation_bias = is_documentation_task(task)

    if not headers:
        if documentation_bias:
            labels.extend(BROAD_DOCUMENTATION_LABELS)
        if product_bias:
            labels.extend(PRODUCT_SECTION_LABELS)

    intent_checks = [
        ("install", ["install", "installation", "setup", "getting started", "quickstart", "deploy", "deployment", "self host", "self-host"]),
        ("configure", ["config", "configuration", "configure", "settings", "environment", "env", "compose", "docker", "yaml", "yml"]),
        ("usage", ["usage", "example", "examples", "how to", "show me how", "how do", "how can", "tutorial", "guide", "build", "create", "develop", "implement", "extension", "plugin"]),
        ("api", ["api", "endpoint", "authentication", "auth", "token", "authorization"]),
        ("troubleshooting", ["troubleshoot", "error", "fix", "issue", "problem", "faq", "known issue"]),
        ("download", ["download", "document", "manual", "pdf", "resource"]),
        ("release", ["release", "changelog", "version", "migration", "upgrade"]),
        ("pricing", ["price", "pricing", "plan", "billing", "cost"]),
        ("security", ["security", "permission", "permissions", "authentication", "authorization"]),
        ("compatibility", ["compatibility", "requirement", "requirements", "supported", "platform"]),
        ("compose", ["docker compose", "docker-compose", "compose file", "compose.yaml", "compose.yml", "docker-compose.yml"]),
        ("extension", ["extension", "extensions", "plugin", "plugins", "sillytavern", "add-on", "addon"]),
        ("schema", ["schema", "manifest", "package", "package.json", "readme", "metadata"]),
        ("specifications", ["spec", "specification", "attribute", "dimension", "thread", "gasket"]),
        ("equipment", ["equipment", "application", "fits", "fitment", "used on", "compatibility"]),
        ("cross_reference", ["cross", "reference", "xref", "interchange", "oem", "oe", "replacement", "replaces", "equivalent", "competitor"]),
        ("maintenance_kits", ["maintenance", "kit", "kits", "service kit", "repair kit"]),
    ]

    for intent, terms in intent_checks:
        if any(term in task_text for term in terms):
            labels.extend(SECTION_INTENT_ALIASES.get(intent, []))

    labels.extend(re.findall(r'"([^"]{3,80})"', task or ""))
    labels.extend(REVEAL_CONTROL_LABELS)

    return unique_preserve_order(labels)


def json_to_text(value: Any, depth: int = 0) -> List[str]:
    if depth > 12:
        return []

    lines = []

    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).strip()
            if key_text.startswith("@"):
                continue

            if isinstance(item, (str, int, float, bool)) and str(item).strip():
                lines.append(f"{key_text}: {item}")
            else:
                child_lines = json_to_text(item, depth + 1)
                if child_lines and key_text:
                    lines.append(f"{key_text}:")
                lines.extend(child_lines)

    elif isinstance(value, list):
        for item in value:
            lines.extend(json_to_text(item, depth + 1))

    elif isinstance(value, (str, int, float, bool)):
        text = str(value).strip()
        if text and len(text) <= 20000:
            lines.append(text)

    return lines


def parse_maybe_json_text(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    try:
        parsed = json.loads(text)
        return "\n".join(json_to_text(parsed))
    except Exception:
        return text


def extract_title_from_html(raw_html: str) -> Optional[str]:
    match = re.search(r"<title[^>]*>(.*?)</title>", raw_html or "", flags=re.I | re.S)
    if not match:
        return None

    title = html.unescape(match.group(1))
    title = re.sub(r"\s+", " ", title).strip()
    return title or None


def extract_json_script_text(raw_html: str) -> str:
    raw_html = raw_html or ""
    extracted = []

    next_data_match = re.search(
        r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
        raw_html,
        flags=re.I | re.S,
    )
    if next_data_match:
        extracted.append(parse_maybe_json_text(html.unescape(next_data_match.group(1))))

    json_script_blocks = re.findall(
        r"<script[^>]+type=[\"']application/(?:ld\+)?json[\"'][^>]*>(.*?)</script>",
        raw_html,
        flags=re.I | re.S,
    )

    for block in json_script_blocks:
        extracted.append(parse_maybe_json_text(html.unescape(block)))

    script_blocks = re.findall(r"<script[^>]*>(.*?)</script>", raw_html, flags=re.I | re.S)

    for block in script_blocks:
        block = html.unescape(block or "").strip()
        if not block:
            continue

        likely_data = any(
            marker in block.lower()
            for marker in [
                "specifications", "oem", "cross reference", "maintenance", "equipment",
                "product", "recordid", "record id", "__next_data__", "__nuxt__", "apollo",
                "redux", "commerce", "salesforce",
            ]
        )

        assignment_matches = re.findall(
            r"(?:window\.[A-Za-z0-9_]+|__INITIAL_STATE__|__APOLLO_STATE__|__NUXT__)\s*=\s*({.*?});",
            block,
            flags=re.S,
        )

        for candidate in assignment_matches:
            extracted.append(parse_maybe_json_text(candidate))

        if likely_data:
            cleaned = re.sub(r"\s+", " ", block).strip()
            if cleaned and len(cleaned) < 100000:
                extracted.append(cleaned)

    return "\n".join(line for line in unique_preserve_order(extracted) if line)


def html_to_text(raw_html: str) -> str:
    raw_html = raw_html or ""
    title = extract_title_from_html(raw_html)
    json_text = extract_json_script_text(raw_html)

    cleaned = re.sub(r"<!--.*?-->", " ", raw_html, flags=re.S)
    cleaned = re.sub(r"<script\b[^>]*>.*?</script>", " ", cleaned, flags=re.I | re.S)
    cleaned = re.sub(r"<style\b[^>]*>.*?</style>", " ", cleaned, flags=re.I | re.S)
    cleaned = re.sub(r"<noscript\b[^>]*>.*?</noscript>", " ", cleaned, flags=re.I | re.S)
    cleaned = re.sub(r"</(h1|h2|h3|h4|h5|h6|tr|li|p|div|section|article|dt|dd|td|th)>", "\n", cleaned, flags=re.I)
    cleaned = re.sub(r"<br\s*/?>", "\n", cleaned, flags=re.I)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)

    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()

    parts = []
    if title:
        parts.append(f"Title: {title}")
    if cleaned:
        parts.append(cleaned)
    if json_text:
        parts.append("Embedded structured data:\n" + json_text)

    return "\n\n".join(parts).strip()


def lineify_text(text: str) -> List[str]:
    text = html.unescape(text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]{2,}", " ", text)

    raw_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        pieces = re.split(r"\s{3,}|\t+| \| ", line)
        for piece in pieces:
            piece = piece.strip(" -\u2022")
            if piece:
                raw_lines.append(piece)

    return unique_preserve_order(raw_lines)


def extract_table_like_rows(text: str, task: Optional[str] = None, max_rows: int = 10000) -> List[str]:
    lines = lineify_text(text)
    task_terms = [term.lower() for term in re.findall(r"[a-zA-Z0-9_\-\.]{3,}", task or "")]

    row_like = []
    for line in lines:
        lower = line.lower()
        has_delimiters = bool(re.search(r"\s{2,}|\||,|\t", line))
        has_year = bool(re.search(r"\b(19|20)\d{2}\b", line))
        has_part_number = bool(re.search(r"\b[A-Z0-9][A-Z0-9\-]{3,}\b", line))
        has_task_term = any(term in lower for term in task_terms)

        if has_delimiters or has_year or has_part_number or has_task_term:
            row_like.append(line)

        if len(row_like) >= max_rows:
            break

    return unique_preserve_order(row_like)


def build_section_alias_map(headers: List[str]) -> Dict[str, str]:
    alias_to_header = {}

    for header in headers:
        normalized = normalize_heading(header)
        aliases = {header, normalized}

        for alias_group in SECTION_INTENT_ALIASES.values():
            normalized_group = {normalize_heading(alias) for alias in alias_group}
            if normalized in normalized_group:
                aliases.update(alias_group)

        for alias in aliases:
            alias_to_header[normalize_heading(alias)] = header

    return alias_to_header


def extract_sections_from_text(text: str, headers: List[str]) -> Dict[str, Dict[str, Any]]:
    lines = lineify_text(text)
    alias_to_header = build_section_alias_map(headers)

    all_stop_headers = set(COMMON_STOP_HEADERS)
    all_stop_headers.update(alias_to_header.keys())

    sections = {header: {"found": False, "content": "", "items": []} for header in headers}

    current_header = None
    current_lines = []

    def flush_current() -> None:
        nonlocal current_header, current_lines

        if not current_header:
            current_lines = []
            return

        content_lines = [line for line in current_lines if normalize_heading(line) != normalize_heading(current_header)]
        content = "\n".join(content_lines).strip()
        sections[current_header] = {
            "found": bool(content),
            "content": content,
            "items": content_lines,
        }

        current_header = None
        current_lines = []

    for line in lines:
        normalized = normalize_heading(line)
        matched_header = alias_to_header.get(normalized)

        if not matched_header:
            for alias, original in alias_to_header.items():
                if normalized == alias or normalized.startswith(alias + " "):
                    if len(normalized) <= len(alias) + 50:
                        matched_header = original
                        break

        if matched_header:
            flush_current()
            current_header = matched_header
            current_lines = []
            continue

        if current_header and normalized in all_stop_headers:
            flush_current()
            continue

        if current_header:
            current_lines.append(line)

    flush_current()

    for header in headers:
        if sections[header]["found"]:
            continue

        aliases = [alias for alias, owner in alias_to_header.items() if owner == header]
        for alias in aliases:
            pattern = re.compile(
                rf"({re.escape(alias)}\s*[:\-]?\s*)(.*?)(?=\n[A-Z][A-Za-z0-9 /&,\-\(\)]{{2,60}}\s*[:\-]?\n|\Z)",
                flags=re.I | re.S,
            )
            match = pattern.search(text)
            if not match:
                continue

            content = match.group(2).strip()[:120000]
            items = lineify_text(content)
            if items:
                sections[header] = {
                    "found": True,
                    "content": "\n".join(items),
                    "items": items,
                }
                break

    return sections


def extract_relevant_lines(text: str, task: str, max_lines: int = 180) -> List[str]:
    lines = lineify_text(text)
    terms = [term.lower() for term in re.findall(r"[a-zA-Z0-9_\-\.]{3,}", task or "")]
    labels = [normalize_heading(label) for label in infer_page_labels(task=task, product_bias=is_product_task(task))]

    scored = []
    for index, line in enumerate(lines):
        lower = line.lower()
        normalized = normalize_heading(line)
        score = 0

        for term in terms:
            if term in lower:
                score += 2

        for label in labels:
            if label and (label in normalized or normalized in label):
                score += 3

        if score:
            context_start = max(0, index - 2)
            context_end = min(len(lines), index + 8)
            scored.append((score, index, lines[context_start:context_end]))

    scored.sort(key=lambda item: item[0], reverse=True)

    output = []
    for _, _, context in scored:
        for line in context:
            if line not in output:
                output.append(line)
            if len(output) >= max_lines:
                return output

    return output[:max_lines]


def extraction_sufficient(task: str, result: Dict[str, Any]) -> bool:
    relevant_lines = result.get("relevant_lines", [])
    found_sections = result.get("found_sections", {})
    table_like_rows = result.get("table_like_rows", [])
    network_count = result.get("network_response_count", 0)
    content_chars = result.get("content_chars", 0)

    task_lower = (task or "").lower()
    wants_table = any(term in task_lower for term in ["table", "rows", "csv", "all equipment", "complete equipment", "list all"])
    wants_product_data = is_product_task(task)

    if wants_table and len(table_like_rows) >= 20:
        return True

    if wants_product_data and (found_sections or len(table_like_rows) >= 10 or len(relevant_lines) >= 10):
        return True

    if found_sections:
        return True

    if len(relevant_lines) >= 8:
        return True

    if network_count > 0 and content_chars > 1000:
        return True

    return content_chars > 3000


def estimate_confidence(result: Dict[str, Any]) -> str:
    if result.get("found_sections") and result.get("network_response_count", 0) > 0:
        return "high"

    if result.get("found_sections") or result.get("table_like_row_count", 0) >= 20:
        return "medium_high"

    if result.get("relevant_lines") or result.get("content_chars", 0) > 3000:
        return "medium"

    if result.get("content_chars", 0) > 500:
        return "low"

    return "very_low"
