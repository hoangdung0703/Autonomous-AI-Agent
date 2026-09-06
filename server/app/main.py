"""FastAPI app entrypoint.

Importing app.config validates all required environment variables at
startup so misconfiguration fails fast, before any request is served.
No routes are registered yet — that is added in a later step.
"""

from fastapi import FastAPI

from app.config import settings  # noqa: F401 — import validates env vars on startup

app = FastAPI(title="Autonomous AI Agent for Enterprise Knowledge Retrieval")
