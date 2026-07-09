"""
main.py — FastAPI application entrypoint for HomeIntel.

Start with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from api.chat import router as chat_router
from api.files import router as files_router
from api.status import router as status_router
from api.transcribe import router as transcribe_router
from ingestion.watcher import start_watcher

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(levelname)-8s %(name)s — %(message)s",
)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("docling").setLevel(logging.WARNING)
logging.getLogger("fastembed").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.skip_llm_health_check:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{settings.ollama_base_url_str}/api/tags")
                if resp.status_code == 200:
                    logger.info(
                        "Ollama connectivity OK — LLM=%s embed=%s",
                        settings.ollama_llm_model,
                        settings.ollama_embed_model,
                    )
                else:
                    logger.warning("Ollama returned HTTP %d at startup", resp.status_code)
        except Exception as exc:
            logger.warning("Ollama not reachable at startup: %s", exc)

    start_watcher()
    yield
    # watcher is a daemon thread — exits with the process


app = FastAPI(
    title="HomeIntel",
    description="Local AI assistant for your NAS — queries your files using RAG.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(files_router)
app.include_router(status_router)
app.include_router(transcribe_router)

# Serve the built frontend (same-origin) when a build exists. Mounted after the
# routers so API routes take priority; guarded so the API still boots without a
# frontend build. Absolute path keeps it working regardless of CWD.
_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
