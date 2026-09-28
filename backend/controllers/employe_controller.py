"""Annuaire par id. Jointure employes_features.id = employes_cible.id."""
from typing import Any, Dict, Optional

from fastapi import HTTPException
from sqlalchemy import text

from database import engine


def lister_annuaire(q: Optional[str] = None, limit: int = 40) -> dict:
    limite = max(1, min(int(limit), 100))
    sql = """
        SELECT f.id, c.a_quitte_l_entreprise
        FROM employes_features f
        LEFT JOIN employes_cible c ON c.id = f.id
    """
    params: Dict[str, Any] = {"limite": limite}
    if q and str(q).strip():
        sql += " WHERE CAST(f.id AS TEXT) LIKE :q "
        params["q"] = f"{str(q).strip()}%"
    sql += " ORDER BY f.id ASC LIMIT :limite"
    try:
        with engine.connect() as conn:
            lignes = conn.execute(text(sql), params).mappings().all()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    personnes = [
        {
            "id": row["id"],
            "libelle": f"Employé #{row['id']}",
            "a_quitte_l_entreprise": row.get("a_quitte_l_entreprise"),
        }
        for row in lignes
    ]
    return {"nb": len(personnes), "personnes": personnes}


def obtenir_employe(identifiant: int) -> dict:
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
