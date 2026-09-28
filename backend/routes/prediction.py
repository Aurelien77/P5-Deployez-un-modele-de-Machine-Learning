from fastapi import APIRouter

from controllers import modele_controller, resultat_controller
from schemas.prediction import PredictionRequest, SauvegardeRequest

router = APIRouter(tags=["prediction"])


@router.get("/colonnes")
def get_colonnes(modele: str):
    return modele_controller.obtenir_colonnes(modele)


@router.post("/predict")
def predict(data: PredictionRequest):
    sortie = modele_controller.predire(data.modele, data.features, data.seuil)
    libelle = "Quitte" if sortie["prediction"] == 1 else "Reste"
    enregistrement = resultat_controller.sauvegarder_prediction(
        SauvegardeRequest(
            prenom=data.prenom or "John",
            nom=data.nom or "Doe",
            modele_utilise=sortie["modele_utilise"],
            probabilite_de_quitter=sortie["probabilite"],
            prediction=sortie["prediction"],
            libelle_prediction=libelle,
            seuil_applique=sortie["seuil_utilise"],
            features=data.features,
        )
    )
    sortie["enregistrement"] = enregistrement
    return sortie
