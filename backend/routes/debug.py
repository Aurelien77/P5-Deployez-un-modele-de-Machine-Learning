from fastapi import APIRouter, Path

from controllers import modele_controller

router = APIRouter(tags=["debug"])


@router.get(
    "/debug/modele/{modele}",
    summary="Inspecter un modèle",
    description="Décrit le modèle chargé : type, scaler, colonnes, seuil et rôles des colonnes.",
)
def debug_modele(
    modele: str = Path(description="Nom du modèle chargé, par exemple `top1`."),
):
    """Inspecte un modèle présent en mémoire. Renvoie 404 s'il n'est pas chargé."""
    return modele_controller.debug_modele(modele)
