"""
Research MCP Server
Combines SearXNG + Crawl4AI (full) + BGE Reranker + Qdrant
Designed for internal Docker networking (no port mapping needed)
"""

import os
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import asyncio

app = FastAPI(title="Research MCP Server")

# ==================== CONFIG ====================
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8080")
CRAWL4AI_URL = os.getenv("CRAWL4AI_URL", "http://crawl4ai:11235")
RERANKER_URL = os.getenv("RERANKER_URL", "http://reranker:8000")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")

qdrant = QdrantClient(url=QDRANT_URL)
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# ==================== MODELS ====================
class SearchRequest(BaseModel):
    query: str
    num_results: int = 10
    deep_crawl: bool = False
    max_crawl_depth: int = 2

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_k: int = 5

class CrawlRequest(BaseModel):
    url: str
    config: Optional[dict] = None

# ==================== HELPER FUNCTIONS ====================
async def search_searxng(query: str, num_results: int = 10):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SEARXNG_URL}/search",
            json={"q": query, "format": "json", "pageno": 1}
        )
        data = response.json()
        return [{"title": r.get("title"), "url": r.get("url"), "content": r.get("content", "")} 
                for r in data.get("results", [])[:num_results]]

async def rerank_results(query: str, documents: List[str], top_k: int = 5):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{RERANKER_URL}/rerank",
            json={"query": query, "docs": documents, "top_k": top_k}
        )
        return response.json().get("results", [])

async def crawl_page(url: str, config: Optional[dict] = None):
    async with httpx.AsyncClient(timeout=120.0) as client:
        payload = {"url": url}
        if config:
            payload["config"] = config
        response = await client.post(f"{CRAWL4AI_URL}/crawl", json=payload)
        return response.json()

# ==================== MCP TOOLS ====================
@app.post("/mcp/smart_search")
async def smart_search(request: SearchRequest):
    """Smart search: SearXNG + Reranker + optional deep crawl"""
    results = await search_searxng(request.query, request.num_results)
    
    # Rerank
    docs = [r["content"] or r["title"] for r in results]
    reranked = await rerank_results(request.query, docs, top_k=request.num_results)
    
    final_results = []
    for i, item in enumerate(reranked):
        final_results.append({
            "title": results[i]["title"],
            "url": results[i]["url"],
            "content": item.get("text", ""),
            "score": item.get("score", 0)
        })
    
    # Optional deep crawl
    if request.deep_crawl:
        for result in final_results[:3]:  # Crawl top 3
            try:
                crawl_result = await crawl_page(result["url"])
                result["full_content"] = crawl_result.get("markdown", "")
            except:
                pass
    
    return {"results": final_results}

@app.post("/mcp/deep_crawl")
async def deep_crawl(request: CrawlRequest):
    """Full Crawl4AI crawl with advanced configuration"""
    result = await crawl_page(request.url, request.config)
    return result

@app.post("/mcp/rerank")
async def rerank(request: RerankRequest):
    """Rerank any list of documents"""
    return await rerank_results(request.query, request.documents, request.top_k)

@app.post("/mcp/hybrid_search")
async def hybrid_search(query: str, limit: int = 10):
    """Search both web and personal knowledge base (Qdrant)"""
    # Web search
    web_results = await search_searxng(query, limit)
    
    # Qdrant search
    query_embedding = embedding_model.encode(query).tolist()
    qdrant_results = qdrant.search(
        collection_name="knowledge_base",
        query_vector=query_embedding,
        limit=limit
    )
    
    return {
        "web_results": web_results,
        "knowledge_base_results": [
            {"content": hit.payload.get("content", ""), "score": hit.score}
            for hit in qdrant_results
        ]
    }

@app.post("/mcp/save_to_knowledge_base")
async def save_to_knowledge_base(url: str, content: str, metadata: Optional[dict] = None):
    """Save content to Qdrant knowledge base"""
    embedding = embedding_model.encode(content).tolist()
    
    qdrant.upsert(
        collection_name="knowledge_base",
        points=[{
            "id": hash(url),
            "vector": embedding,
            "payload": {
                "url": url,
                "content": content,
                "metadata": metadata or {}
            }
        }]
    )
    return {"status": "saved", "url": url}

# ==================== HEALTH CHECK ====================
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "services": {
            "searxng": SEARXNG_URL,
            "crawl4ai": CRAWL4AI_URL,
            "reranker": RERANKER_URL,
            "qdrant": QDRANT_URL
        }
    }
