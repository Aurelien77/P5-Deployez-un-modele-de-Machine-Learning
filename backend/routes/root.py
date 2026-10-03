from fastapi import APIRouter

from controllers import root_controller

router = APIRouter(tags=["accueil"])


@router.get(
    "/",
    summary="Accueil",
    description="Sert `index.html` s'il est présent dans le dossier de travail, sinon renvoie 404.",
)
def read_root():
    """Sert la page d'accueil ou signale que `index.html` est absent."""
    return root_controller.servir_accueil()
