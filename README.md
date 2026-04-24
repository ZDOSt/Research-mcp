# Research MCP Server

A powerful, unified MCP server that combines:
- **SearXNG** (meta-search)
- **Crawl4AI** (full features)
- **BGE Reranker**
- **Qdrant** (vector knowledge base)

## Features

### Available MCP Tools

| Tool                  | Description                                      |
|-----------------------|--------------------------------------------------|
| `smart_search`        | SearXNG + Reranker + optional deep crawl         |
| `deep_crawl`          | Full Crawl4AI with advanced configuration        |
| `rerank`              | Rerank any list of documents                     |
| `hybrid_search`       | Search web + personal knowledge base (Qdrant)    |
| `save_to_knowledge_base` | Save content to Qdrant for future retrieval   |

## Setup

1. Place this folder in your project
2. Add the `research-mcp` service to your main `docker-compose.yml`
3. Rebuild:

```bash
docker compose up -d --build research-mcp
```

## MCP Connection (LibreChat / Claude / Cursor)

**SSE Endpoint:**
```
http://research-mcp:8000/mcp/sse
```

**Or use the REST API directly** at `http://research-mcp:8010`

## Internal Docker Networking

This server is designed to work with **internal Docker service names** only.
No port mapping is required in production.

## Health Check

```
GET /health
```

## Future Improvements

- Add more advanced Crawl4AI strategies
- Add memory / conversation context
- Add webhooks for async jobs
