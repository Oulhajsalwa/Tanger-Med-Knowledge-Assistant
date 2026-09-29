
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Tanger Med Knowledge Assistant API",
        description=(
            "API RAG pour l'exploration des rapports annuels et RSE du Groupe "
            "Tanger Med, avec citations et gestion de l'abstention."
        ),
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {"service": "tanger-med-rag", "docs": "/docs", "health": "/api/health"}

    return app


app = create_app()
