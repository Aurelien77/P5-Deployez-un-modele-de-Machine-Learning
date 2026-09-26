from fastapi import APIRouter

from controllers import root_controller

router = APIRouter(tags=["accueil"])


@router.get("/")
def read_root():
    return root_controller.servir_accueil()
