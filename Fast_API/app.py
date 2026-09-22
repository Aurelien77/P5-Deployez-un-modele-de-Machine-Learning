import os
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

from sqlalchemy import create_engine, Column, Integer, String, Float, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --- CONFIGURATION DE LA BASE DE DONNÉES POSTGRESQL ---
DATABASE_URL = "postgresql+psycopg://postgres:mysecretpassword@localhost:5432/rh_predictions_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ResultatDB(Base):
    __tablename__ = "resultats"

    id = Column(Integer, primary_key=True, index=True)
    prenom = Column(String, default="John")
    nom = Column(String, default="Doe")
    modele_utilise = Column(String)
    probabilite_de_quitter = Column(Float)
    prediction = Column(Integer)
    libelle_prediction = Column(String)
    seuil_applique = Column(Float)
    details = Column(String)


Base.metadata.create_all(bind=engine)


def _assurer_colonne_details():
    try:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE resultats ADD COLUMN IF NOT EXISTS details TEXT"))
            conn.commit()
    except Exception as exc:
        print(f"Note migration colonne details : {exc}")


_assurer_colonne_details()

# --- INITIALISATION FASTAPI ---
app = FastAPI(title="API Prédiction RH & Sauvegarde")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Colonnes réellement attendues par le Pipeline LogisticRegression
COLONNES_MODELE = [
    "age",
    "revenu_mensuel",
    "annees_dans_l_entreprise",
    "distance_domicile_travail",
    "annes_sous_responsable_actuel",
    "satisfaction_employee_environnement",
    "niveau_hierarchique_poste",
    "satisfaction_min",
    "poste_x_niveau",
    "formations_par_an",
    "statut_marital",
    "frequence_deplacement",
    "heure_supplementaires",
    "hs_et_salaire_bas",
    "jeune_faible_anciennete",
]

COLONNES_PAR_DEFAUT = COLONNES_MODELE.copy()

CLES_MODELE_POSSIBLES = ["model", "modele", "estimator", "clf", "classifier", "pipeline", "best_model", "best_estimator"]
CLES_SCALER_POSSIBLES = ["scaler", "preprocessor", "preprocesseur", "transformer", "scaler_x", "encoder"]
CLES_COLONNES_POSSIBLES = ["columns", "colonnes", "features", "feature_names", "feature_names_in_", "X_columns"]
CLES_SEUIL_POSSIBLES = ["threshold", "seuil", "best_threshold", "optimal_threshold"]


def _nom_type(obj) -> str:
    return type(obj).__name__


def trouver_column_transformers(modele) -> list:
    trouves = []
    if modele is None:
        return trouves
    if _nom_type(modele) == "ColumnTransformer":
        trouves.append(modele)
    if hasattr(modele, "named_steps"):
        for etape in modele.named_steps.values():
            trouves.extend(trouver_column_transformers(etape))
    if hasattr(modele, "transformers_"):
        for _nom, trans, _cols in modele.transformers_:
            if trans is None or trans == "drop" or trans == "passthrough":
                continue
            trouves.extend(trouver_column_transformers(trans))
    return trouves


def extraire_colonnes_par_role(modele) -> dict:
    """Lit le ColumnTransformer pour savoir quelles colonnes sont one-hot vs numériques."""
    roles = {"onehot": [], "numeriques": [], "autres": [], "detail": []}
    for ct in trouver_column_transformers(modele):
        for nom, trans, cols in getattr(ct, "transformers_", []):
            if cols is None or cols == "remainder" or cols == "drop":
                continue
            try:
                cols_liste = list(cols)
            except TypeError:
                continue

            etapes = []
            if hasattr(trans, "named_steps"):
                etapes = list(trans.named_steps.values())
            else:
                etapes = [trans]

            noms_etapes = [_nom_type(e) for e in etapes]
            func_repr = None
            for e in etapes:
                if _nom_type(e) == "FunctionTransformer":
                    func_repr = repr(getattr(e, "func", None))

            roles["detail"].append({
                "branche": nom,
                "etapes": noms_etapes,
                "colonnes": cols_liste,
                "function_transformer": func_repr,
            })

            if any("OneHot" in n for n in noms_etapes):
                roles["onehot"].extend(cols_liste)
            elif any(n in ("StandardScaler", "MinMaxScaler", "RobustScaler", "SimpleImputer", "FunctionTransformer") for n in noms_etapes):
                roles["numeriques"].extend(cols_liste)
            else:
                roles["autres"].extend(cols_liste)

    # dédoublonne en conservant l'ordre
    for cle in ("onehot", "numeriques", "autres"):
        vus = set()
        propres = []
        for col in roles[cle]:
            if col not in vus:
                vus.add(col)
                propres.append(col)
        roles[cle] = propres
    return roles


def extraire_colonnes_du_modele(modele) -> List[str]:
    """Essaie d'extraire feature_names_in_ d'un estimateur scikit-learn (brut, Pipeline, etc.)."""
    if modele is None:
        return []
    if hasattr(modele, "feature_names_in_"):
        cols = list(modele.feature_names_in_)
        if cols:
            return cols
    if hasattr(modele, "named_steps"):
        for etape in modele.named_steps.values():
            if hasattr(etape, "feature_names_in_"):
                cols = list(etape.feature_names_in_)
                if cols:
                    return cols
    return []


def analyser_objet_charge(obj, nom_modele: str) -> dict:
    resultat = {"model": None, "scaler": None, "colonnes": [], "seuil": None, "type_objet": str(type(obj))}

    if hasattr(obj, "predict"):
        resultat["model"] = obj
        resultat["colonnes"] = extraire_colonnes_du_modele(obj)
        print(f"[{nom_modele}] Objet chargé = estimateur scikit-learn direct.")
        return resultat

    if isinstance(obj, dict):
        print(f"[{nom_modele}] Objet chargé = dict. Clés disponibles : {list(obj.keys())}")

        for cle in CLES_MODELE_POSSIBLES:
            if cle in obj and hasattr(obj[cle], "predict"):
                resultat["model"] = obj[cle]
                print(f"[{nom_modele}] Modèle trouvé sous la clé '{cle}' -> {type(obj[cle])}")
                break

        if resultat["model"] is None:
            for cle, valeur in obj.items():
                if hasattr(valeur, "predict"):
                    resultat["model"] = valeur
                    print(f"[{nom_modele}] Modèle trouvé par introspection sous la clé '{cle}' -> {type(valeur)}")
                    break

        for cle in CLES_SCALER_POSSIBLES:
            if cle in obj and hasattr(obj[cle], "transform") and obj[cle] is not resultat["model"]:
                resultat["scaler"] = obj[cle]
                print(f"[{nom_modele}] Scaler trouvé sous la clé '{cle}' -> {type(obj[cle])}")
                break

        for cle in CLES_COLONNES_POSSIBLES:
            if cle in obj:
                valeur = obj[cle]
                try:
                    cols = list(valeur)
                    if cols and all(isinstance(c, str) for c in cols):
                        resultat["colonnes"] = cols
                        print(f"[{nom_modele}] Colonnes trouvées sous la clé '{cle}' ({len(cols)} colonnes)")
                        break
                except TypeError:
                    pass

        if not resultat["colonnes"] and resultat["model"] is not None:
            resultat["colonnes"] = extraire_colonnes_du_modele(resultat["model"])

        for cle in CLES_SEUIL_POSSIBLES:
            if cle in obj and isinstance(obj[cle], (int, float)):
                resultat["seuil"] = float(obj[cle])
                print(f"[{nom_modele}] Seuil trouvé sous la clé '{cle}' : {resultat['seuil']}")
                break

        if resultat["model"] is None:
            print(f"[{nom_modele}] ATTENTION : aucun objet avec .predict trouvé dans le dict.")

        return resultat

    print(f"[{nom_modele}] ATTENTION : type d'objet non reconnu ({type(obj)}), ni estimateur ni dict.")
    return resultat


COLONNES_CATEGORIELLES = [
    "statut_marital",
    "frequence_deplacement",
    "heure_supplementaires",
]

ENCODAGES_CATEGORIELS = {
    "heure_supplementaires": {
        "non": 0, "no": 0, "0": 0, "false": 0,
        "oui": 1, "yes": 1, "1": 1, "true": 1,
    },
    "frequence_deplacement": {
        "aucun": 0, "aucune": 0, "non-travel": 0, "nontravel": 0, "0": 0,
        "occasionnel": 1, "travel_rarely": 1, "travelrarely": 1, "rare": 1, "1": 1,
        "frequent": 2, "fréquent": 2, "travel_frequently": 2, "travelfrequently": 2, "2": 2,
    },
    "statut_marital": {
        "célibataire": 0, "celibataire": 0, "single": 0, "0": 0,
        "marié(e)": 1, "marie(e)": 1, "marié": 1, "marie": 1, "married": 1, "1": 1,
        "divorcé(e)": 2, "divorce(e)": 2, "divorcé": 2, "divorce": 2, "divorced": 2, "2": 2,
    },
}


def _to_float(valeur, defaut=0.0) -> float:
    try:
        if valeur is None or valeur == "":
            return defaut
        if isinstance(valeur, str):
            valeur = valeur.replace(",", ".")
        return float(valeur)
    except (TypeError, ValueError):
        return defaut


def _normaliser_texte(valeur: Any) -> str:
    return str(valeur).strip().lower()


def encoder_valeur_categorielle(colonne: str, valeur: Any) -> float:
    if valeur is None or valeur == "":
        return 0.0
    if isinstance(valeur, (int, float)) and not isinstance(valeur, bool):
        return float(valeur)

    table = ENCODAGES_CATEGORIELS.get(colonne, {})
    code = table.get(_normaliser_texte(valeur))
    if code is None:
        raise HTTPException(
            status_code=422,
            detail=f"Valeur inconnue pour '{colonne}' : {valeur}. "
                   f"Valeurs attendues : {sorted(set(table.keys()))}",
        )
    return float(code)


def completer_features_calculees(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calcule les 2 features d'ingénierie si elles sont absentes.
    - formations_par_an = nb_formations_suivies / annee_experience_totale
    - poste_x_niveau    = niveau_hierarchique_poste * niveau_education
    """
    f = dict(features)

    if "formations_par_an" not in f or f.get("formations_par_an") in (None, ""):
        exp = _to_float(f.get("annee_experience_totale"))
        nb_form = _to_float(f.get("nb_formations_suivies"))
        f["formations_par_an"] = (nb_form / exp) if exp else 0.0

    if "poste_x_niveau" not in f or f.get("poste_x_niveau") in (None, ""):
        niveau_poste = _to_float(f.get("niveau_hierarchique_poste"))
        niveau_educ = _to_float(f.get("niveau_education"))
        f["poste_x_niveau"] = niveau_poste * niveau_educ

    f["formations_par_an"] = _to_float(f.get("formations_par_an"))
    f["poste_x_niveau"] = _to_float(f.get("poste_x_niveau"))
    return f


def preparer_dataframe_numerique(features: Dict[str, Any], colonnes: List[str]) -> pd.DataFrame:
    """DataFrame 100% numérique, dans l'ordre exact des colonnes du modèle."""
    f = completer_features_calculees(features)
    ligne = {}
    for col in colonnes:
        if col in COLONNES_CATEGORIELLES:
            ligne[col] = encoder_valeur_categorielle(col, f.get(col))
        else:
            ligne[col] = _to_float(f.get(col))
    df = pd.DataFrame([ligne], columns=colonnes)
    return df.astype(np.float64)


def preparer_dataframe_mixte(features: Dict[str, Any], colonnes: List[str]) -> pd.DataFrame:
    """Numérique + catégories en texte, pour un ColumnTransformer avec OneHotEncoder."""
    f = completer_features_calculees(features)
    ligne = {}
    for col in colonnes:
        if col in COLONNES_CATEGORIELLES:
            valeur = f.get(col, "")
            ligne[col] = "" if valeur is None else str(valeur)
        else:
            ligne[col] = _to_float(f.get(col))
    return pd.DataFrame([ligne], columns=colonnes)


def preparer_dataframe_aligne(features: Dict[str, Any], colonnes: List[str], roles: dict) -> pd.DataFrame:
    """
    Aligne les dtypes sur le ColumnTransformer réel du .pkl :
    - colonnes OneHotEncoder -> str
    - colonnes StandardScaler / FunctionTransformer -> float
    """
    f = completer_features_calculees(features)
    onehot = set(roles.get("onehot") or COLONNES_CATEGORIELLES)
    ligne = {}
    for col in colonnes:
        if col in onehot:
            valeur = f.get(col, "")
            ligne[col] = "" if valeur is None else str(valeur)
        else:
            if col in COLONNES_CATEGORIELLES:
                ligne[col] = encoder_valeur_categorielle(col, f.get(col))
            else:
                ligne[col] = _to_float(f.get(col))
    df = pd.DataFrame([ligne], columns=colonnes)
    for col in colonnes:
        if col in onehot:
            df[col] = df[col].astype("object")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)
    return df


def _proba_depuis_prediction(sortie) -> float:
    tableau = np.asarray(sortie)
    if tableau.ndim == 2 and tableau.shape[1] >= 2:
        return float(tableau[0][1])
    return float(tableau.ravel()[0])


def appeler_modele(model, df_num: pd.DataFrame, df_mixte: pd.DataFrame, df_aligne: pd.DataFrame, scaler=None):
    """Essaie les formats compatibles avec un ColumnTransformer (DataFrame obligatoire)."""
    candidats = [
        ("dataframe_aligne", df_aligne),
        ("dataframe_mixte", df_mixte),
        ("dataframe_float64", df_num),
    ]

    erreurs = []
    for nom, entree in candidats:
        try:
            X = entree
            if scaler is not None:
                X = scaler.transform(entree)
            if hasattr(model, "predict_proba"):
                return _proba_depuis_prediction(model.predict_proba(X)), nom
            return _proba_depuis_prediction(model.predict(X)), nom
        except Exception as exc:
            erreurs.append(f"{nom}: {type(exc).__name__}: {exc}")

    raise RuntimeError("Aucun format d'entrée n'a fonctionné. Détails : " + " | ".join(erreurs))


# --- CHARGEMENT DES MODÈLES MACHINE LEARNING (Dossier 'modeles/') ---
MODELS = {
    "top1": "modeles/LogisticRegression.pkl",
}

loaded_models = {}

for nom_modele, chemin in MODELS.items():
    if os.path.exists(chemin):
        try:
            obj_brut = joblib.load(chemin)
            print(f"\n--- Chargement modèle '{nom_modele}' ---")
            info = analyser_objet_charge(obj_brut, nom_modele)

            if info["model"] is not None:
                if not info["colonnes"]:
                    info["colonnes"] = COLONNES_MODELE.copy()
                if info["seuil"] is None:
                    info["seuil"] = 0.37
                info["roles"] = extraire_colonnes_par_role(info["model"])
                loaded_models[nom_modele] = info
                print(
                    f"[{nom_modele}] -> {len(info['colonnes'])} colonnes, "
                    f"scaler={'oui' if info['scaler'] is not None else 'non'}, "
                    f"seuil_sauve={info['seuil']}"
                )
                print(f"[{nom_modele}] Structure ColumnTransformer : {info['roles']['detail']}")
                print(f"[{nom_modele}] Colonnes OneHot : {info['roles']['onehot']}")
                print(f"[{nom_modele}] Colonnes numériques : {info['roles']['numeriques']}")
            else:
                print(f"[{nom_modele}] Échec : impossible d'identifier un modèle utilisable dans le fichier.")
            print("--- Fin chargement ---\n")

        except Exception as e:
            print(f"Erreur lors du chargement du modèle {nom_modele}: {e}")
    else:
        print(f"Attention : Le fichier {chemin} est introuvable.")


class PredictionRequest(BaseModel):
    modele: str
    features: Dict[str, Any]
    seuil: float = 0.37


class SauvegardeRequest(BaseModel):
    prenom: Optional[str] = "John"
    nom: Optional[str] = "Doe"
    modele_utilise: str
    probabilite_de_quitter: float
    prediction: int
    libelle_prediction: str
    seuil_applique: float
    features: Dict[str, Any] = {}


@app.get("/")
def read_root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    raise HTTPException(status_code=404, detail="Fichier index.html introuvable dans le dossier du projet.")


@app.get("/colonnes")
def get_colonnes(modele: str):
    if modele not in loaded_models:
        return {"colonnes": COLONNES_PAR_DEFAUT, "source": "defaut_modele_non_charge", "seuil": 0.37}

    cols = loaded_models[modele]["colonnes"] or COLONNES_PAR_DEFAUT
    return {
        "colonnes": cols,
        "source": "extraction_automatique" if loaded_models[modele]["colonnes"] else "defaut_extraction_echouee",
        "seuil": loaded_models[modele]["seuil"] if loaded_models[modele]["seuil"] is not None else 0.37,
    }


@app.get("/debug/modele/{modele}")
def debug_modele(modele: str):
    if modele not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Modèle '{modele}' non chargé ou non identifiable.")

    info = loaded_models[modele]
    return {
        "type_model": str(type(info["model"])),
        "a_scaler": info["scaler"] is not None,
        "type_scaler": str(type(info["scaler"])) if info["scaler"] is not None else None,
        "nb_colonnes": len(info["colonnes"]),
        "colonnes": info["colonnes"],
        "seuil_sauvegarde": info["seuil"],
        "roles": info.get("roles", {}),
    }


@app.post("/predict")
def predict(data: PredictionRequest):
    if data.modele not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Modèle '{data.modele}' introuvable ou non chargé.")

    info = loaded_models[data.modele]
    model = info["model"]
    scaler = info["scaler"]
    colonnes = info["colonnes"] or COLONNES_MODELE

    try:
        features_completes = completer_features_calculees(data.features)
        roles = info.get("roles") or extraire_colonnes_par_role(model)
        df_num = preparer_dataframe_numerique(features_completes, colonnes)
        df_mixte = preparer_dataframe_mixte(features_completes, colonnes)
        df_aligne = preparer_dataframe_aligne(features_completes, colonnes, roles)

        proba, format_utilise = appeler_modele(
            model, df_num, df_mixte, df_aligne, scaler=scaler
        )
        prediction_finale = 1 if proba >= data.seuil else 0

        return {
            "modele_utilise": data.modele,
            "probabilite": proba,
            "prediction": prediction_finale,
            "seuil_utilise": data.seuil,
            "colonnes_utilisees": colonnes,
            "format_entree": format_utilise,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")


@app.post("/sauvegarder")
def sauvegarder_prediction(data: SauvegardeRequest):
    db = SessionLocal()
    try:
        nouveau_resultat = ResultatDB(
            prenom=data.prenom,
            nom=data.nom,
            modele_utilise=data.modele_utilise,
            probabilite_de_quitter=data.probabilite_de_quitter,
            prediction=data.prediction,
            libelle_prediction=data.libelle_prediction,
            seuil_applique=data.seuil_applique,
            details=json.dumps(data.features, ensure_ascii=False),
        )
        db.add(nouveau_resultat)
        db.commit()
        db.refresh(nouveau_resultat)

        return {
            "message": "Enregistré avec succès en base !",
            "id": nouveau_resultat.id,
            "employe": f"{nouveau_resultat.prenom} {nouveau_resultat.nom}",
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/resultats")
def lister_resultats(limit: int = 20):
    db = SessionLocal()
    try:
        lignes = (
            db.query(ResultatDB)
            .order_by(ResultatDB.id.desc())
            .limit(max(1, min(limit, 100)))
            .all()
        )
        sortie = []
        for row in lignes:
            details = {}
            if row.details:
                try:
                    details = json.loads(row.details)
                except json.JSONDecodeError:
                    details = {"brut": row.details}
            sortie.append({
                "id": row.id,
                "prenom": row.prenom,
                "nom": row.nom,
                "modele_utilise": row.modele_utilise,
                "probabilite_de_quitter": row.probabilite_de_quitter,
                "prediction": row.prediction,
                "libelle_prediction": row.libelle_prediction,
                "seuil_applique": row.seuil_applique,
                "details": details,
            })
        return {"nb": len(sortie), "resultats": sortie}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
