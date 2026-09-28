import json
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import text

from database import SessionLocal, engine
from models.resultat import ResultatDB
from schemas.prediction import SauvegardeRequest


def sauvegarder_prediction(data: SauvegardeRequest) -> dict:
    db = SessionLocal()
    try:
        details = dict(data.features or {})
        if getattr(data, "employe_id", None) is not None:
            details["employe_id"] = data.employe_id
        champs = {
            "prenom": data.prenom,
            "nom": data.nom,
            "modele_utilise": data.modele_utilise,
            "probabilite_de_quitter": data.probabilite_de_quitter,
            "prediction": data.prediction,
            "libelle_prediction": data.libelle_prediction,
            "seuil_applique": data.seuil_applique,
            "details": json.dumps(details, ensure_ascii=False),
        }
        if hasattr(ResultatDB, "employe_id"):
            champs["employe_id"] = getattr(data, "employe_id", None)
        nouveau_resultat = ResultatDB(**champs)
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


def lister_annuaire(q: Optional[str] = None, limit: int = 2000) -> dict:
    """Tous les id de employes_features (jointure cible)."""
    return lister_annuaire_par_id(q=q, limit=limit)


def lister_resultats(
    limit: int = 20,
    prenom: Optional[str] = None,
    nom: Optional[str] = None,
) -> dict:
    db = SessionLocal()
    try:
        requete = db.query(ResultatDB)
        if prenom:
            requete = requete.filter(ResultatDB.prenom.ilike(prenom.strip()))
        if nom:
            requete = requete.filter(ResultatDB.nom.ilike(nom.strip()))
        lignes = (
            requete.order_by(ResultatDB.id.desc())
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


def lister_annuaire_par_id(q: Optional[str] = None, limit: int = 40) -> dict:
    limite = max(1, min(int(limit or 2000), 5000))
    sql = """
        SELECT f.id, c.a_quitte_l_entreprise
        FROM employes_features f
        LEFT JOIN employes_cible c ON c.id = f.id
    """
    params = {"limite": limite}
    if q and str(q).strip():
        sql += " WHERE CAST(f.id AS TEXT) LIKE :q "
        params["q"] = f"{str(q).strip()}%"
    sql += " ORDER BY f.id ASC LIMIT :limite"
    try:
        with engine.connect() as conn:
            lignes = conn.execute(text(sql), params).mappings().all()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "nb": len(lignes),
        "personnes": [
            {
                "id": row["id"],
                "libelle": f"Employé #{row['id']}",
                "a_quitte_l_entreprise": row.get("a_quitte_l_entreprise"),
            }
            for row in lignes
        ],
    }


def obtenir_employe_par_id(identifiant: int) -> dict:
    sql = """
        SELECT f.*, c.a_quitte_l_entreprise
        FROM employes_features f
        LEFT JOIN employes_cible c ON c.id = f.id
        WHERE f.id = :identifiant
    """
    try:
        with engine.connect() as conn:
            row = conn.execute(text(sql), {"identifiant": identifiant}).mappings().first()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if row is None:
        raise HTTPException(status_code=404, detail=f"Employé #{identifiant} introuvable.")
    return {"employe": dict(row)}
