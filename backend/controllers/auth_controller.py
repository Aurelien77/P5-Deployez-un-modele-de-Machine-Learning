from fastapi import HTTPException

from auth_utils import creer_token, hacher_mot_de_passe, lire_token, verifier_mot_de_passe
from database import SessionLocal
from models.resultat import ResultatDB
from models.user import UserDB


def creer_compte(username: str, password: str) -> dict:
    identifiant = username.strip()
    db = SessionLocal()
    try:
        existant = db.query(UserDB).filter(UserDB.username == identifiant).first()
        if existant is not None:
            raise HTTPException(status_code=409, detail="Ce nom d'utilisateur existe déjà.")
        utilisateur = UserDB(username=identifiant, password_hash=hacher_mot_de_passe(password))
        db.add(utilisateur)
        db.commit()
        db.refresh(utilisateur)
        return creer_token(utilisateur.id, utilisateur.username)
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        db.close()


def connecter(username: str, password: str) -> dict:
    identifiant = username.strip()
    db = SessionLocal()
    try:
        utilisateur = db.query(UserDB).filter(UserDB.username == identifiant).first()
        if utilisateur is None:
            raise HTTPException(status_code=404, detail="Cet utilisateur n'existe pas.")
        if not verifier_mot_de_passe(password, utilisateur.password_hash):
            raise HTTPException(status_code=401, detail="Mot de passe incorrect.")
        return creer_token(utilisateur.id, utilisateur.username)
    finally:
        db.close()


def supprimer_compte(user_id: int) -> dict:
    db = SessionLocal()
    try:
        utilisateur = db.query(UserDB).filter(UserDB.id == user_id).first()
        if utilisateur is None:
            raise HTTPException(status_code=404, detail="Compte introuvable.")
        db.query(ResultatDB).filter(ResultatDB.id_user == user_id).delete(synchronize_session=False)
        db.delete(utilisateur)
        db.commit()
        return {"message": "Compte et prédictions associées supprimés."}
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        db.close()


def utilisateur_depuis_token(token: str) -> dict:
    try:
        payload = lire_token(token)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    db = SessionLocal()
    try:
        utilisateur = db.query(UserDB).filter(UserDB.id == payload["id"]).first()
        if utilisateur is None:
            raise HTTPException(status_code=401, detail="Compte introuvable ou supprimé.")
        return {"id": utilisateur.id, "username": utilisateur.username}
    finally:
        db.close()


def changer_mot_de_passe(user_id: int, ancien_password: str, nouveau_password: str) -> dict:
    db = SessionLocal()
    try:
        utilisateur = db.query(UserDB).filter(UserDB.id == user_id).first()
        if utilisateur is None:
            raise HTTPException(status_code=404, detail="Compte introuvable.")
        if not verifier_mot_de_passe(ancien_password, utilisateur.password_hash):
            raise HTTPException(status_code=401, detail="Ancien mot de passe incorrect.")
        utilisateur.password_hash = hacher_mot_de_passe(nouveau_password)
        db.commit()
        return {"message": "Mot de passe modifié."}
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        db.close()
