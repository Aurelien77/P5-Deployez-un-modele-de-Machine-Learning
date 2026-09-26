from fastapi import APIRouter

from controllers import modele_controller
from schemas.prediction import PredictionRequest

router = APIRouter(tags=["prediction"])


@router.get("/colonnes")
def get_colonnes(modele: str):
    return modele_controller.obtenir_colonnes(modele)


@router.post("/predict")
def predict(data: PredictionRequest):
    return modele_controller.predire(data.modele, data.features, data.seuil)
