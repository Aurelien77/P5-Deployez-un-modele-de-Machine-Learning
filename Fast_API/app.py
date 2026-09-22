import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional

# --- IMPORTS SQLALCHEMY ---
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --- CONFIGURATION DE LA BASE DE DONNÉES POSTGRESQL ---
DATABASE_URL = "postgresql://postgres:mysecretpassword@localhost:5433/rh_predictions_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Modèle SQLAlchemy pour la table 'resultats' avec probabilite_de_quitter et libelle_prediction
class ResultatDB(Base):
    __tablename__ = "resultats"

    id = Column(Integer, primary_key=True, index=True)
    prenom = Column(String, default="John")
    nom = Column(String, default="Doe")
    modele_utilise = Column(String)
    probabilite_de_quitter = Column(Float)  # <-- Renommé ici
    prediction = Column(Integer)
    libelle_prediction = Column(String)  # "Reste" ou "Quitte"
    seuil_applique = Column(Float)

# Création automatique des tables si elles n'existent pas
Base.metadata.create_all(bind=engine)

# --- INITIALISATION FASTAPI ---
app = FastAPI(title="API Prédiction RH & Sauvegarde")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CHARGEMENT DES MODÈLES MACHINE LEARNING (Dossier 'modeles/') ---
MODELS = {
    "top1": "modeles/LogisticRegression.pkl",
 
}

loaded_models = {}
for nom_modele, chemin in MODELS.items():
    if os.path.exists(chemin):
        try:
            loaded_models[nom_modele] = joblib.load(chemin)
        except Exception as e:
            print(f"Erreur lors du chargement du modèle {nom_modele}: {e}")
    else:
        print(f"Attention : Le fichier {chemin} est introuvable.")

# --- SCHÉMAS PYDANTIC ---
class PredictionRequest(BaseModel):
    modele: str
    features: Dict[str, Any]
    seuil: float = 0.5

class SauvegardeRequest(BaseModel):
    prenom: Optional[str] = "John"
    nom: Optional[str] = "Doe"
    modele_utilise: str
    probabilite_de_quitter: float  # <-- Renommé ici
    prediction: int
    libelle_prediction: str
    seuil_applique: float
    features: Dict[str, Any] = {}


# --- ROUTES DE L'API ---

@app.get("/")
def read_root():
    """Sert directement l'interface graphique HTML."""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    raise HTTPException(status_code=404, detail="Fichier index.html introuvable dans le dossier du projet.")


@app.get("/colonnes")
def get_colonnes(modele: str):
    """Renvoie la liste des features attendues par le modèle sélectionné."""
    if modele not in loaded_models:
        colonnes_defaut = [
            "age", "revenu_mensuel", "nombre_experiences_precedentes", "annee_experience_totale",
            "annees_dans_l_entreprise", "annees_dans_le_poste_actuel", "nombre_participation_pee",
            "nb_formations_suivies", "distance_domicile_travail", "niveau_education",
            "annees_depuis_la_derniere_promotion", "annes_sous_responsable_actuel",
            "satisfaction_employee_environnement", "note_evaluation_precedente",
            "niveau_hierarchique_poste", "satisfaction_employee_nature_travail",
            "satisfaction_employee_equipe", "satisfaction_employee_equilibre_pro_perso",
            "augentation_salaire_precedente", "satisfaction_globale", "satisfaction_min",
            "montant_augmentation_precedente", "ratio_anciennete_carriere", "inertie_poste",
            "stagnation_promotion", "revenu_par_annee_experience", "revenu_par_niveau",
            "delta_performance", "hs_et_salaire_bas", "jeune_faible_anciennete", "trajet_long",
            "job_hopper", "genre", "statut_marital", "departement", "poste", "domaine_etude",
            "frequence_deplacement", "heure_supplementaires"
        ]
        return {"colonnes": colonnes_defaut}
    
    m = loaded_models[modele]
    if hasattr(m, "feature_names_in_"):
        colonnes = list(m.feature_names_in_)
    elif hasattr(m, "named_steps") and hasattr(list(m.named_steps.values())[-1], "feature_names_in_"):
        colonnes = list(list(m.named_steps.values())[-1].feature_names_in_)
    else:
        colonnes = list(m.feature_names_in_) if hasattr(m, "feature_names_in_") else []
        
    return {"colonnes": colonnes}


@app.post("/predict")
def predict(data: PredictionRequest):
    """Effectue la prédiction en fonction du modèle, des features et du seuil."""
    if data.modele not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Modèle '{data.modele}' introuvable ou non chargé.")
    
    model = loaded_models[data.modele]
    
    try:
        df_input = pd.DataFrame([data.features])
        
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(df_input)[0][1])
        else:
            pred_brute = model.predict(df_input)[0]
            proba = float(pred_brute)

        prediction_finale = 1 if proba >= data.seuil else 0

        return {
            "modele_utilise": data.modele,
            "probabilite": proba,
            "prediction": prediction_finale,
            "seuil_utilise": data.seuil
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")


@app.post("/sauvegarder")
def sauvegarder_prediction(data: SauvegardeRequest):
    """Enregistre la prédiction, le libellé texte, et les métadonnées dans PostgreSQL."""
    db = SessionLocal()
    try:
        nouveau_resultat = ResultatDB(
            prenom=data.prenom,
            nom=data.nom,
            modele_utilise=data.modele_utilise,
            probabilite_de_quitter=data.probabilite_de_quitter,  # <-- Enregistrement avec le nouveau nom
            prediction=data.prediction,
            libelle_prediction=data.libelle_prediction,
            seuil_applique=data.seuil_applique
        )
        db.add(nouveau_resultat)
        db.commit()
        db.refresh(nouveau_resultat)
        
        return {
            "message": "Enregistré avec succès en base !",
            "id": nouveau_resultat.id,
            "employe": f"{nouveau_resultat.prenom} {nouveau_resultat.nom}"
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()