from fastapi import APIRouter, Depends, Query

from controllers import modele_controller, resultat_controller
from routes.auth import utilisateur_courant
from schemas.prediction import PredictionRequest, SauvegardeRequest

router = APIRouter(tags=["prediction"])


@router.get(
    "/colonnes",
    summary="Colonnes du modèle",
    description=(
        "Liste les colonnes attendues par un modèle chargé, avec le seuil associé. "
        "Si le modèle est inconnu, renvoie les colonnes par défaut."
    ),
)
def get_colonnes(
    modele: str = Query(description="Nom du modèle chargé, par exemple `top1`."),
):
    """Retourne les colonnes et le seuil du modèle demandé."""
    return modele_controller.obtenir_colonnes(modele)


@router.post(
    "/predict",
    summary="Prédire et enregistrer",
    description=(
        "Calcule la probabilité de départ, applique le seuil, puis enregistre le résultat "
        "pour l'utilisateur connecté (jeton Bearer, 12 h). "
        "Une prédiction de 1 signifie un risque de départ."
    ),
)
def predict(
    data: PredictionRequest,
    utilisateur: dict = Depends(utilisateur_courant),
):
    """Prédit le risque de départ et sauvegarde la prédiction de l'utilisateur."""
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
            employe_id=data.employe_id,
        ),
        id_user=utilisateur["id"],
    )
    sortie["enregistrement"] = enregistrement
    return sortie
