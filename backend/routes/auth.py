from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from controllers import auth_controller
from schemas.user import CompteRequest, MotDePasseRequest

#Ajoute le préfixe auth/ devant les routes pour organiser nominativement les routes.

router = APIRouter(prefix="/auth", tags=["auth"])

#recupère le token dans l'entête http

_bearer = HTTPBearer(auto_error=False)

# Est-ce que un token de connexion existe sinon répond 401, non authtentifié. 

def utilisateur_courant(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict:
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Connexion requise (jeton Bearer).")
    return auth_controller.utilisateur_depuis_token(credentials.credentials)

#Debut des routes

@router.get("/sante", summary="Vérifier que les routes auth sont chargées")
def sante():
    return {"ok": True, "service": "auth"}


@router.post("/register", summary="Créer un compte", status_code=201)
def register(data: CompteRequest):
    """Crée un utilisateur et renvoie un jeton valable 12 h."""
    return auth_controller.creer_compte(data.username, data.password)


@router.post("/login", summary="Se connecter")
def login(data: CompteRequest):
    """Vérifie le mot de passe haché et renvoie un jeton valable 12 h."""
    return auth_controller.connecter(data.username, data.password)


@router.get("/moi", summary="Utilisateur courant")
def moi(utilisateur: dict = Depends(utilisateur_courant)):
    return utilisateur


@router.delete("/compte", summary="Supprimer son compte")
def supprimer_compte(utilisateur: dict = Depends(utilisateur_courant)):
    """Supprime le compte et les prédictions liées (id_user)."""
    return auth_controller.supprimer_compte(utilisateur["id"])


@router.post("/mot-de-passe", summary="Modifier son mot de passe")
def changer_mot_de_passe(
    data: MotDePasseRequest,
    utilisateur: dict = Depends(utilisateur_courant),
):
    """Vérifie l'ancien mot de passe, puis enregistre le nouveau haché."""
    return auth_controller.changer_mot_de_passe(
        utilisateur["id"], data.ancien_password, data.nouveau_password
    )
