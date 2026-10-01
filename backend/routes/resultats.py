from typing import Optional

from fastapi import APIRouter

from controllers import resultat_controller
from schemas.prediction import SauvegardeRequest

router = APIRouter(tags=["resultats"])


@router.post("/sauvegarder")
def sauvegarder_prediction(data: SauvegardeRequest):
    return resultat_controller.sauvegarder_prediction(data)


@router.get("/resultats")
def lister_resultats(
    limit: int = 20,
    prenom: Optional[str] = None,
    nom: Optional[str] = None,
):
    return resultat_controller.lister_resultats(limit, prenom=prenom, nom=nom)

@router.get("/resultats/{resultat_id}")
def detail_resultat(resultat_id: int):
    return resultat_controller.obtenir_detail_resultat(resultat_id)

@router.get("/annuaire")
def lister_annuaire(q: Optional[str] = None, limit: int = 2000):
    return resultat_controller.lister_annuaire(q=q, limit=limit)


@router.get("/employes/{identifiant}")
def obtenir_employe(identifiant: int):
    return resultat_controller.obtenir_employe_par_id(identifiant)
