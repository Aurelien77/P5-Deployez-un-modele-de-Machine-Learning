from typing import Any, Dict, Optional

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    modele: str
    features: Dict[str, Any]
    seuil: float = 0.37


class SauvegardeRequest(BaseModel):
    prenom: Optional[str] = "John"
    nom: Optional[str] = "Doe"
    modele_utilise: str
    probabilite_de_quitter: float
    prediction: int
    libelle_prediction: str
    seuil_applique: float
    features: Dict[str, Any] = {}
