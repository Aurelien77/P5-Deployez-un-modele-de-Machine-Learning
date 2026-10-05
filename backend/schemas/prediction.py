from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

#Validation des données de prédiction et de sauvegarde via pydantic. 
# Ces classes sont utilisées dans les routes FastAPI pour valider les requêtes entrantes et générer la documentation OpenAPI.


class PredictionRequest(BaseModel):
    """Demande de prédiction. Le résultat est calculé puis enregistré."""

    modele: str = Field(description="Nom du modèle chargé, par exemple `top1`.")
    features: Dict[str, Any] = Field(description="Valeurs des colonnes du modèle.")
    seuil: float = Field(0.37, description="Seuil au-dessus duquel l'employé est classé à risque.")
    prenom: Optional[str] = Field("John", description="Prénom enregistré avec la prédiction.")
    nom: Optional[str] = Field("Doe", description="Nom enregistré avec la prédiction.")
    employe_id: Optional[int] = Field(
        None,
        description="Identifiant annuaire. Vide si les features sont saisies à la main.",
    )


class SauvegardeRequest(BaseModel):
    """Prédiction déjà calculée, à enregistrer dans `resultats`."""

    prenom: Optional[str] = Field("John", description="Prénom associé à la prédiction.")
    nom: Optional[str] = Field("Doe", description="Nom associé à la prédiction.")
    modele_utilise: str = Field(description="Modèle qui a produit la prédiction.")
    probabilite_de_quitter: float = Field(description="Probabilité de départ, entre 0 et 1.")
    prediction: int = Field(description="1 si le seuil est dépassé, sinon 0.")
    libelle_prediction: str = Field(description="Libellé lisible, par exemple Quitte ou Reste.")
    seuil_applique: float = Field(description="Seuil utilisé pour trancher la prédiction.")
    features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Features de la saisie. Ignorées en base si `employe_id` est renseigné.",
    )
    employe_id: Optional[int] = Field(
        None,
        description="Employé lié. Doit exister dans `employes_features` si renseigné.",
    )
