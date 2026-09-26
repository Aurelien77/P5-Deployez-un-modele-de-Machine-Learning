from fastapi import APIRouter

from controllers import resultat_controller
from schemas.prediction import SauvegardeRequest

router = APIRouter(tags=["resultats"])


@router.post("/sauvegarder")
def sauvegarder_prediction(data: SauvegardeRequest):
    return resultat_controller.sauvegarder_prediction(data)


@router.get("/resultats")
def lister_resultats(limit: int = 20):
    return resultat_controller.lister_resultats(limit)
