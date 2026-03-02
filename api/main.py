"""
FastAPI application factory for the Financial Risk Intelligence Copilot.

Sets up the application with CORS, middleware, routers,
and lifespan events for resource initialization/cleanup.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is in sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.middleware import RequestLoggingMiddleware, register_exception_handlers
from api.routes import fraud, health, rag, reasoning
from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)
config = Config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Startup: Pre-loads heavy resources (model, indices)
    Shutdown: Cleanup and logging
    """
    logger.info("=" * 60)
    logger.info("RiskIntellect-AI — Starting up")
    logger.info("=" * 60)

    # Pre-load components (non-blocking — failures are logged, not fatal)
    try:
        from api.dependencies import get_fraud_predictor
        get_fraud_predictor()
    except Exception as e:
        logger.warning(f"ML model pre-load skipped: {e}")

    try:
        from api.dependencies import get_hybrid_retriever
        get_hybrid_retriever()
    except Exception as e:
        logger.warning(f"RAG indices pre-load skipped: {e}")

    logger.info("Startup complete — API is ready")

    yield  # Application runs here

    logger.info("RiskIntellect-AI — Shutting down")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance.
    """
    app = FastAPI(
        title="RiskIntellect-AI",
        description=(
            "Made by Ayan — Production-grade AI system combining supervised ML fraud detection, "
            "RAG over regulatory documents, and LLM reasoning for risk assessment."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──
    cors_origins = config.get("api.cors_origins", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request Logging Middleware ──
    app.add_middleware(RequestLoggingMiddleware)

    # ── Exception Handlers ──
    register_exception_handlers(app)

    # ── Routers ──
    app.include_router(health.router)
    app.include_router(fraud.router)
    app.include_router(rag.router)
    app.include_router(reasoning.router)

    # ── Static Files (Frontend Dashboard) ──
    if FRONTEND_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

        @app.get("/dashboard", response_class=HTMLResponse)
        async def serve_dashboard():
            """Serve the frontend dashboard."""
            index_path = FRONTEND_DIR / "index.html"
            return index_path.read_text()

        logger.info(f"Frontend dashboard mounted at /dashboard")

    return app


# Create the app instance (used by uvicorn)
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=config.get("api.host", "0.0.0.0"),
        port=config.get("api.port", 8000),
        reload=config.get("app.debug", False),
        log_level="info",
    )
