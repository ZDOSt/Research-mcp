from fastapi import FastAPI

from shared import (
    COLLECTION_NAME,
    IngestRequest,
    QueryRequest,
    SourceDeleteRequest,
    SourceListRequest,
    delete_source_impl,
    init_qdrant,
    list_sources_impl,
    qdrant,
    rag_ingest_impl,
    rag_query_impl,
    source_stats_impl,
)

app = FastAPI(title="Research RAG API")

init_qdrant()


@app.get("/health")
async def health():
    try:
        qdrant.get_collection(COLLECTION_NAME)
        qdrant_status = "ok"
    except Exception:
        qdrant_status = "error"

    return {"status": "ok", "qdrant": qdrant_status}


@app.get("/rag/health")
async def rag_health():
    return await health()


@app.post("/rag/ingest")
async def rag_ingest_route(body: IngestRequest):
    return await rag_ingest_impl(body)


@app.post("/rag/query")
async def rag_query_route(body: QueryRequest):
    return await rag_query_impl(body)


@app.post("/rag/sources")
async def list_sources_route(body: SourceListRequest):
    return await list_sources_impl(limit=body.limit)


@app.get("/rag/source-stats")
async def source_stats_route():
    return await source_stats_impl()


@app.post("/rag/delete-source")
async def delete_source_route(body: SourceDeleteRequest):
    return await delete_source_impl(body.source)
