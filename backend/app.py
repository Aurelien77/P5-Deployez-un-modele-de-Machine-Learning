"""Point d'entrée FastAPI.

Les routes sont déclarées dans `routes/` et déléguent toute la logique métier
aux controllers. Ce module reste volontairement mince pour rester compatible
avec :
- Docker : `uvicorn app:app`
- les tests : `from backend.app import app, COLONNES_MODELE`
"""
import os
import sys

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from controllers.modele_controller import COLONNES_MODELE, charger_modeles
from database import initialiser_base
from models.resultat import ResultatDB  # noqa: F401 — enregistre le modèle sur Base
from routes import register_routes

initialiser_base()
charger_modeles()

app = FastAPI(title="API Prédiction RH & Sauvegarde")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_routes(app)

__all__ = ["app", "COLONNES_MODELE"]
