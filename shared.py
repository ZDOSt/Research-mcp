import hashlib
import logging
import os
import re
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from fastapi import HTTPException
from fastembed import TextEmbedding
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8080")
CRAWL4AI_URL = os.getenv("CRAWL4AI_URL", "http://crawl4ai:11235")
RERANKER_URL = os.getenv("RERANKER_URL", "http://reranker:8000")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "librechat_docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./models")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1100"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "384"))
QDRANT_INIT_RETRIES = int(os.getenv("QDRANT_INIT_RETRIES", "30"))
QDRANT_INIT_DELAY_SECONDS = float(os.getenv("QDRANT_INIT_DELAY_SECONDS", "2"))

qdrant = QdrantClient(url=QDRANT_URL)
embedder = TextEmbedding(model_name=EMBEDDING_MODEL, cache_dir=MODEL_CACHE_DIR)


class IngestRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class SourceDeleteRequest(BaseModel):
    source: str


class SourceListRequest(BaseModel):
    limit: int = Field(default=50, ge=1, le=500)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def runtime_retrieval_context() -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "retrieved_at_utc": now.isoformat(),
        "current_date_utc": now.date().isoformat(),
        "freshness": "runtime_retrieved",
        "guidance": (
            "This MCP result was retrieved or queried at server runtime. "
            "Information dated after the answering model's training cutoff can be valid; "
            "do not discard it solely because it is newer than the model cutoff."
        ),
    }


def normalize_url(url: str) -> str:
    return (url or "").strip()


def get_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        return domain[4:] if domain.startswith("www.") else domain
    except Exception:
        return ""


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _looks_not_found(exc: Exception) -> bool:
    message = str(exc).lower()
    return "not found" in message or "404" in message


def _looks_already_exists(exc: Exception) -> bool:
    message = str(exc).lower()
    return "already exists" in message or "409" in message


def init_qdrant() -> None:
    last_error = None

    for attempt in range(1, QDRANT_INIT_RETRIES + 1):
        try:
            qdrant.get_collection(COLLECTION_NAME)
            logger.info("Collection '%s' exists.", COLLECTION_NAME)
            return
        except Exception as exc:
            last_error = exc

            if _looks_not_found(exc):
                try:
                    qdrant.create_collection(
                        collection_name=COLLECTION_NAME,
                        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
                    )
                    logger.info("Created collection '%s'.", COLLECTION_NAME)
                    return
                except Exception as create_exc:
                    if _looks_already_exists(create_exc):
                        logger.info("Collection '%s' was created by another process.", COLLECTION_NAME)
                        return
                    last_error = create_exc

            logger.warning(
                "Qdrant init attempt %s/%s failed: %s",
                attempt,
                QDRANT_INIT_RETRIES,
                last_error,
            )
            if attempt < QDRANT_INIT_RETRIES:
                time.sleep(QDRANT_INIT_DELAY_SECONDS)

    raise RuntimeError(f"Qdrant initialization failed after {QDRANT_INIT_RETRIES} attempts: {last_error}")


def clean_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_markdown_sections(text: str) -> List[Dict[str, str]]:
    text = clean_text(text)
    if not text:
        return []

    lines = text.splitlines()
    sections = []
    current_heading = "Document"
    current_lines = []

    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    for line in lines:
        match = heading_re.match(line)
        if match and current_lines:
            sections.append(
                {
                    "heading": current_heading,
                    "text": "\n".join(current_lines).strip(),
                }
            )
            current_heading = match.group(2).strip()
            current_lines = [line]
        elif match:
            current_heading = match.group(2).strip()
            current_lines.append(line)
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(
            {
                "heading": current_heading,
                "text": "\n".join(current_lines).strip(),
            }
        )

    return [section for section in sections if section["text"]]


def split_long_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)

        if end < len(text):
            sentence_boundary = max(
                text.rfind(". ", start, end),
                text.rfind("? ", start, end),
                text.rfind("! ", start, end),
                text.rfind("\n\n", start, end),
            )

            if sentence_boundary > start + int(chunk_size * 0.55):
                end = sentence_boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def chunk_text_with_metadata(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    sections = split_markdown_sections(text)
    chunks = []

    for section_index, section in enumerate(sections):
        section_chunks = split_long_text(section["text"], chunk_size, overlap)

        for section_chunk_index, chunk in enumerate(section_chunks):
            chunks.append(
                {
                    "text": chunk,
                    "section": section["heading"],
                    "section_index": section_index,
                    "section_chunk_index": section_chunk_index,
                }
            )

    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    return [list(vec) for vec in embedder.embed(texts)]


def point_id_for(source: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{chunk_index}"))


def qdrant_query_points(query_vec: List[float], limit: int):
    if hasattr(qdrant, "query_points"):
        response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return getattr(response, "points", response)

    return qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=limit,
    )


async def rerank_docs(query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    if not docs:
        return []

    texts = [doc["text"] for doc in docs]

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{RERANKER_URL}/rerank",
                json={"query": query, "texts": texts},
            )
            resp.raise_for_status()
            payload = resp.json()

        scored_docs = []

        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue

                index = item.get("index")
                score = item.get("score", 0)

                if isinstance(index, int) and 0 <= index < len(docs):
                    doc = dict(docs[index])
                    doc["rerank_score"] = score
                    scored_docs.append(doc)

        elif isinstance(payload, dict) and isinstance(payload.get("results"), list):
            for item in payload["results"]:
                if not isinstance(item, dict):
                    continue

                index = item.get("index")
                score = item.get("score", 0)
                text = item.get("text")

                if isinstance(index, int) and 0 <= index < len(docs):
                    doc = dict(docs[index])
                    doc["rerank_score"] = score
                    scored_docs.append(doc)
                elif text:
                    for doc in docs:
                        if doc["text"] == text:
                            ranked_doc = dict(doc)
                            ranked_doc["rerank_score"] = score
                            scored_docs.append(ranked_doc)
                            break

        if scored_docs:
            scored_docs.sort(key=lambda item: item.get("rerank_score", 0), reverse=True)
            return scored_docs[:top_k]

    except Exception as exc:
        logger.warning("Reranker failed, using vector order: %s", str(exc))

    return docs[:top_k]


async def rag_ingest_impl(req: IngestRequest) -> Dict[str, Any]:
    try:
        text = clean_text(req.text)
        if not text:
            return {"stored": 0}

        metadata = req.metadata or {}

        source = normalize_url(metadata.get("source") or metadata.get("url") or "unknown")
        url = normalize_url(metadata.get("url") or source)
        title = metadata.get("title")
        domain = metadata.get("domain") or get_domain(url)
        query = metadata.get("query")
        content_type = metadata.get("content_type", "webpage")
        source_score = metadata.get("source_score")
        source_reason = metadata.get("source_reason")
        retrieved_at_utc = metadata.get("retrieved_at_utc")
        retrieval_current_date_utc = metadata.get("retrieval_current_date_utc")
        ingested_at = utc_now_iso()

        chunks = chunk_text_with_metadata(text)

        if not chunks:
            return {"stored": 0, "source": source}

        vectors = embed_texts([chunk["text"] for chunk in chunks])

        points = []

        for index, (chunk, vec) in enumerate(zip(chunks, vectors)):
            chunk_text = chunk["text"]
            chunk_hash = hash_text(chunk_text)

            payload = {
                "text": chunk_text,
                "source": source,
                "url": url,
                "domain": domain,
                "hash": chunk_hash,
                "chunk_index": index,
                "section": chunk.get("section"),
                "section_index": chunk.get("section_index"),
                "section_chunk_index": chunk.get("section_chunk_index"),
                "content_type": content_type,
                "ingested_at": ingested_at,
                "retrieved_at_utc": retrieved_at_utc,
                "retrieval_current_date_utc": retrieval_current_date_utc,
            }

            if title:
                payload["title"] = title
            if query:
                payload["query"] = query
            if source_score is not None:
                payload["source_score"] = source_score
            if source_reason:
                payload["source_reason"] = source_reason

            points.append(
                PointStruct(
                    id=point_id_for(source, index),
                    vector=vec,
                    payload=payload,
                )
            )

        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.info("Ingested %d chunks from %s.", len(points), source)

        return {
            "stored": len(points),
            "source": source,
            "url": url,
            "title": title,
            "domain": domain,
            "ingested_at": ingested_at,
        }

    except Exception as exc:
        logger.error("Ingest failed: %s", str(exc))
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}")


async def rag_query_impl(req: QueryRequest) -> Dict[str, Any]:
    try:
        top_k = max(1, min(req.top_k, 30))
        query_vec = embed_texts([req.query])[0]

        hits = qdrant_query_points(query_vec=query_vec, limit=top_k * 5)

        unique_docs = []
        seen_text = set()

        for hit in hits:
            payload = hit.payload or {}
            text = payload.get("text", "")

            if not text or text in seen_text:
                continue

            doc = {
                "text": text,
                "source": payload.get("source", "unknown"),
                "url": payload.get("url", payload.get("source", "unknown")),
                "title": payload.get("title"),
                "domain": payload.get("domain"),
                "section": payload.get("section"),
                "chunk_index": payload.get("chunk_index"),
                "content_type": payload.get("content_type"),
                "ingested_at": payload.get("ingested_at"),
                "retrieved_at_utc": payload.get("retrieved_at_utc"),
                "retrieval_current_date_utc": payload.get("retrieval_current_date_utc"),
                "source_score": payload.get("source_score"),
                "source_reason": payload.get("source_reason"),
                "vector_score": getattr(hit, "score", None),
            }

            unique_docs.append(doc)
            seen_text.add(text)

        if not unique_docs:
            return {"query": req.query, "results": []}

        final = await rerank_docs(req.query, unique_docs, top_k)

        return {
            "query": req.query,
            "results": final,
        }

    except Exception as exc:
        logger.error("Query failed: %s", str(exc))
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")


def scroll_points(limit_per_page: int = 10000, max_points: int = 100000):
    offset = None
    total = 0

    while total < max_points:
        points, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=min(limit_per_page, max_points - total),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        total += len(points)
        yield from points

        if offset is None:
            break


async def list_sources_impl(limit: int = 50) -> Dict[str, Any]:
    limit = max(1, min(limit, 500))

    try:
        sources = {}

        for point in scroll_points():
            payload = point.payload or {}
            source = payload.get("source") or payload.get("url") or "unknown"

            if source not in sources:
                sources[source] = {
                    "source": source,
                    "url": payload.get("url", source),
                    "title": payload.get("title"),
                    "domain": payload.get("domain"),
                    "content_type": payload.get("content_type"),
                    "ingested_at": payload.get("ingested_at"),
                    "chunks": 0,
                }

            sources[source]["chunks"] += 1

            current_ingested = sources[source].get("ingested_at")
            new_ingested = payload.get("ingested_at")
            if new_ingested and (not current_ingested or new_ingested > current_ingested):
                sources[source]["ingested_at"] = new_ingested

        sorted_sources = sorted(
            sources.values(),
            key=lambda item: item.get("ingested_at") or "",
            reverse=True,
        )

        return {
            "count": len(sorted_sources),
            "sources": sorted_sources[:limit],
        }

    except Exception as exc:
        logger.error("List sources failed: %s", str(exc))
        raise HTTPException(status_code=500, detail=f"List sources failed: {exc}")


async def source_stats_impl() -> Dict[str, Any]:
    try:
        source_counter = Counter()
        domain_counter = Counter()
        content_type_counter = Counter()
        total = 0

        for point in scroll_points():
            total += 1
            payload = point.payload or {}
            source_counter[payload.get("source", "unknown")] += 1
            domain_counter[payload.get("domain", "unknown")] += 1
            content_type_counter[payload.get("content_type", "unknown")] += 1

        return {
            "collection": COLLECTION_NAME,
            "total_chunks_sampled": total,
            "unique_sources": len(source_counter),
            "top_domains": [
                {"domain": domain, "chunks": count}
                for domain, count in domain_counter.most_common(25)
            ],
            "top_sources": [
                {"source": source, "chunks": count}
                for source, count in source_counter.most_common(25)
            ],
            "content_types": [
                {"content_type": content_type, "chunks": count}
                for content_type, count in content_type_counter.most_common()
            ],
        }

    except Exception as exc:
        logger.error("Source stats failed: %s", str(exc))
        raise HTTPException(status_code=500, detail=f"Source stats failed: {exc}")


async def delete_source_impl(source: str) -> Dict[str, Any]:
    source = normalize_url(source)

    if not source:
        raise HTTPException(status_code=400, detail="source is required")

    try:
        delete_filter = Filter(
            should=[
                FieldCondition(key="source", match=MatchValue(value=source)),
                FieldCondition(key="url", match=MatchValue(value=source)),
            ]
        )

        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=delete_filter,
        )

        return {
            "deleted": True,
            "source": source,
        }

    except Exception as exc:
        logger.error("Delete source failed: %s", str(exc))
        raise HTTPException(status_code=500, detail=f"Delete source failed: {exc}")
