"""FastAPI app entrypoint.

Importing app.config validates all required environment variables at
startup so misconfiguration fails fast, before any request is served.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings  # noqa: F401 — import validates env vars on startup
from app.routes import chat, health

app = FastAPI(title="Autonomous AI Agent for Enterprise Knowledge Retrieval")

# Local-only thesis project — Vite dev server origin only. Configurable via
# CORS_ALLOWED_ORIGINS (comma-separated) for flexibility, defaulting to the
# Vite dev server's default port.
_allowed_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api")
app.include_router(health.router, prefix="/api")
