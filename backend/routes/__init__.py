from fastapi import FastAPI

from routes.auth import router as auth_router
from routes.debug import router as debug_router
from routes.prediction import router as prediction_router
from routes.resultats import router as resultats_router
from routes.root import router as root_router


def register_routes(app: FastAPI) -> None:
    app.include_router(root_router)
    app.include_router(auth_router)
    # Deux montages : Nginx peut retirer /api ou le laisser.
    # /auth/register et /api/auth/register répondent tous les deux.
    app.include_router(auth_router, prefix="/api")
    app.include_router(prediction_router)
    app.include_router(resultats_router)
    app.include_router(debug_router)
