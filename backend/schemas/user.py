from pydantic import BaseModel, Field

#Validation des données utilisateur via pydantic.

class CompteRequest(BaseModel):
    """Identifiants pour créer un compte ou se connecter."""

    username: str = Field(min_length=3, max_length=80, description="Nom d'utilisateur unique.")
    password: str = Field(min_length=4, max_length=128, description="Mot de passe (haché en base).")


class MotDePasseRequest(BaseModel):
    """Ancien et nouveau mot de passe."""

    ancien_password: str = Field(min_length=4, max_length=128)
    nouveau_password: str = Field(min_length=4, max_length=128)
