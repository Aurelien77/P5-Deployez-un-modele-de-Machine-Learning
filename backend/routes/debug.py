from fastapi import APIRouter

from controllers import modele_controller

router = APIRouter(tags=["debug"])


@router.get("/debug/modele/{modele}")
def debug_modele(modele: str):
    return modele_controller.debug_modele(modele)
