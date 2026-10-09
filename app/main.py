"""FastAPI application factory and wiring."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import auth, chat, documents
from app.config import get_settings
from app.core.database import init_db
from app.core.rate_limit import limiter
from app.core.vectorstore import get_collection
from app.services import embeddings

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepare storage and (optionally) warm the embedding model at startup."""
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    get_collection()  # create/open the Chroma collection eagerly
    if settings.preload_embeddings:
        logger.info("Loading embedding model %s ...", settings.embedding_model)
        embeddings.get_model()
    logger.info("LLM provider: %s (model: %s)", settings.llm_provider, settings.llm_model)
    if not settings.llm_api_key:
        logger.warning(
            "No API key set for LLM provider %r; /chat will fail when it needs the LLM",
            settings.llm_provider,
        )
    yield


def create_app() -> FastAPI:
    """Build the FastAPI app with routers, rate limiting and lifecycle hooks."""
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.exception_handler(StarletteHTTPException)
    async def log_http_exception(request: Request, exc: StarletteHTTPException) -> Response:
        """Log *why* a request failed (uvicorn only prints the status code)."""
        level = logging.ERROR if exc.status_code >= 500 else logging.WARNING
        logger.log(
            level, "%s %s -> %s: %s", request.method, request.url.path, exc.status_code, exc.detail
        )
        return await http_exception_handler(request, exc)

    @app.exception_handler(RequestValidationError)
    async def log_validation_error(request: Request, exc: RequestValidationError) -> Response:
        """Log which fields failed validation (request bodies are never logged)."""
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        logger.warning("%s %s -> 422: %s", request.method, request.url.path, problems)
        return await request_validation_exception_handler(request, exc)

    app.include_router(auth.router)
    app.include_router(documents.router)
    app.include_router(chat.router)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        """Serve the single-page web UI."""
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    return app


app = create_app()
