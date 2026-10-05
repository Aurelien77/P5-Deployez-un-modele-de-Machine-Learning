import json
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import text

from database import SessionLocal, engine
from models.resultat import ResultatDB
from schemas.prediction import SauvegardeRequest


def sauvegarder_prediction(data: SauvegardeRequest, id_user: Optional[int] = None) -> dict:
    db = SessionLocal()
    try:
        employe_id = getattr(data, "employe_id", None)
        details_json = None if employe_id else json.dumps(data.features or {}, ensure_ascii=False)

        nouveau_resultat = ResultatDB(
            prenom=data.prenom,
            nom=data.nom,
            modele_utilise=data.modele_utilise,
            probabilite_de_quitter=data.probabilite_de_quitter,
            prediction=data.prediction,
            libelle_prediction=data.libelle_prediction,
            seuil_applique=data.seuil_applique,
            employe_id=employe_id,
            details=details_json,
        )
        if id_user is not None and hasattr(ResultatDB, "id_user"):
            nouveau_resultat.id_user = id_user
        db.add(nouveau_resultat)
        db.commit()
        db.refresh(nouveau_resultat)

        return {
            "message": "Enregistré avec succès en base !",
            "id": nouveau_resultat.id,
            "employe": f"{nouveau_resultat.prenom} {nouveau_resultat.nom}",
            "id_user": getattr(nouveau_resultat, "id_user", None),
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

def lister_resultats(limit: int = 20, prenom: Optional[str] = None, nom: Optional[str] = None, id_user: Optional[int] = None) -> dict:
    db = SessionLocal()
    try:
        if id_user is None:
            return {"nb": 0, "resultats": []}
        requete = db.query(ResultatDB).filter(ResultatDB.id_user == id_user)
        if prenom:
            requete = requete.filter(ResultatDB.prenom.ilike(prenom.strip()))
        if nom:
            requete = requete.filter(ResultatDB.nom.ilike(nom.strip()))
        lignes = requete.order_by(ResultatDB.id.desc()).limit(max(1, min(limit, 100))).all()

        sortie = []
        for row in lignes:
            sortie.append({
                "id": row.id,
                "prenom": row.prenom,
                "nom": row.nom,
                "modele_utilise": row.modele_utilise,
                "probabilite_de_quitter": row.probabilite_de_quitter,
                "prediction": row.prediction,
                "libelle_prediction": row.libelle_prediction,
                "seuil_applique": row.seuil_applique,
                "id_user": row.id_user,
                "employe_id": row.employe_id,  # ← AJOUT : le lien, visible direct dans la liste
            })
        return {"nb": len(sortie), "resultats": sortie}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


def obtenir_detail_resultat(resultat_id: int, id_user: Optional[int] = None) -> dict:
    """Renvoie un résultat + ses features, récupérées :
    - via jointure sur employes_features si employe_id est renseigné
    - via le JSON de secours `details` sinon (saisie manuelle)
    """
    db = SessionLocal()
    try:
        row = db.query(ResultatDB).filter(ResultatDB.id == resultat_id).first()
        if row is None or (id_user is not None and row.id_user != id_user):
            raise HTTPException(status_code=404, detail=f"Résultat #{resultat_id} introuvable.")

        if row.employe_id is not None:
            with engine.connect() as conn:
                ligne = conn.execute(
                    text("SELECT * FROM employes_features WHERE id = :id"),
                    {"id": row.employe_id},
                ).mappings().first()
            features = dict(ligne) if ligne else {}
            features.pop("id", None)
            source = "employes_features"
        else:
            features = json.loads(row.details) if row.details else {}
            source = "details_json"

        return {
            "id": row.id,
            "prenom": row.prenom,
            "nom": row.nom,
            "modele_utilise": row.modele_utilise,
            "probabilite_de_quitter": row.probabilite_de_quitter,
            "prediction": row.prediction,
            "libelle_prediction": row.libelle_prediction,
            "seuil_applique": row.seuil_applique,
            "employe_id": row.employe_id,
            "id_user": row.id_user,
            "features": features,
            "source_features": source,
        }
    finally:
        db.close()


def lister_annuaire(q: Optional[str] = None, limit: int = 2000) -> dict:
    return lister_annuaire_par_id(q=q, limit=limit)


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