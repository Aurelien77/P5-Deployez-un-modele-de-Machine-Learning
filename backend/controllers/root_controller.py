import os

from fastapi import HTTPException
from fastapi.responses import FileResponse


def servir_accueil():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    raise HTTPException(
        status_code=404,
        detail="Fichier index.html introuvable dans le dossier du projet.",
    )
