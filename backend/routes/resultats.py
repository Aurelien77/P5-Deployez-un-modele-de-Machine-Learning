from typing import Optional

from fastapi import APIRouter, Depends, Path, Query

from controllers import resultat_controller
from routes.auth import utilisateur_courant
from schemas.prediction import SauvegardeRequest

router = APIRouter(tags=["resultats"])


@router.post(
    "/sauvegarder",
    summary="Enregistrer une prédiction",
    description=(
        "Enregistre une prédiction déjà calculée pour l'utilisateur connecté. "
        "Sans `employe_id`, les features sont copiées dans `details`."
    ),
)
def sauvegarder_prediction(
    data: SauvegardeRequest,
    utilisateur: dict = Depends(utilisateur_courant),
):
    """Enregistre une prédiction dans la table `resultats`, liée à id_user."""
    return resultat_controller.sauvegarder_prediction(data, id_user=utilisateur["id"])


@router.get(
    "/resultats",
    summary="Lister les prédictions",
    description="Historique des prédictions de l'utilisateur connecté, du plus récent au plus ancien.",
)
def lister_resultats(
    limit: int = Query(20, description="Nombre maximum de lignes renvoyées."),
    prenom: Optional[str] = Query(None, description="Filtre exact sur le prénom."),
    nom: Optional[str] = Query(None, description="Filtre exact sur le nom."),
    utilisateur: dict = Depends(utilisateur_courant),
):
    """Liste les prédictions de l'utilisateur connecté."""
    return resultat_controller.lister_resultats(
        limit, prenom=prenom, nom=nom, id_user=utilisateur["id"]
    )


@router.get(
    "/resultats/{resultat_id}",
    summary="Détail d'une prédiction",
    description=(
        "Renvoie une prédiction de l'utilisateur connecté et ses features. "
        "Si `employe_id` est renseigné, les features viennent de `employes_features` ; "
        "sinon elles sont lues dans le JSON `details`."
    ),
)
def detail_resultat(
    resultat_id: int = Path(description="Identifiant de la ligne dans `resultats`."),
    utilisateur: dict = Depends(utilisateur_courant),
):
    """Détaille une prédiction de l'utilisateur et la source de ses features."""
    return resultat_controller.obtenir_detail_resultat(resultat_id, id_user=utilisateur["id"])


@router.get(
    "/annuaire",
    summary="Annuaire des employés",
    description="Liste les employés connus, avec l'indication s'ils ont quitté l'entreprise.",
)
def lister_annuaire(
    q: Optional[str] = Query(None, description="Début de l'identifiant employé à rechercher."),
    limit: int = Query(2000, description="Nombre maximum d'employés renvoyés."),
):
    """Recherche des employés par début d'identifiant."""
    return resultat_controller.lister_annuaire(q=q, limit=limit)


@router.get(
    "/employes/{identifiant}",
    summary="Fiche employé",
    description="Renvoie les features d'un employé et son label réel, s'il existe.",
)
def obtenir_employe(
    identifiant: int = Path(description="Identifiant dans `employes_features`."),
):
    """Retourne la fiche d'un employé de l'annuaire."""
    return resultat_controller.obtenir_employe_par_id(identifiant)
