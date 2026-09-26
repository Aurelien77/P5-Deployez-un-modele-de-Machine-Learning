import json

from fastapi import HTTPException

from database import SessionLocal
from models.resultat import ResultatDB
from schemas.prediction import SauvegardeRequest


def sauvegarder_prediction(data: SauvegardeRequest) -> dict:
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


def lister_resultats(limit: int = 20) -> dict:
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
