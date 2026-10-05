"""Hachage des mots de passe et jetons de session (12 h).

Aucune dépendance externe : PBKDF2-HMAC-SHA256 et un jeton signé HMAC.
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import time

ITERATIONS = 200_000
DUREE_TOKEN_SECONDES = 12 * 60 * 60
AUTH_SECRET = os.getenv("AUTH_SECRET", "dev-secret-change-me")

#Parrie hachage er vérification de mot de passe. 
# Le mot de passe est haché avec PBKDF2-HMAC-SHA256, 200 000 itérations et un sel aléatoire de 16 octets.
#Le mots de passe est récupéré dans le .env a la racine et n'est pas commité.


def hacher_mot_de_passe(mot_de_passe: str) -> str:
    sel = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", mot_de_passe.encode("utf-8"), sel, ITERATIONS)
    return f"pbkdf2_sha256${ITERATIONS}${sel.hex()}${digest.hex()}"


def verifier_mot_de_passe(mot_de_passe: str, empreinte: str) -> bool:
    try:
        algo, iterations, sel_hex, digest_hex = empreinte.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        sel = bytes.fromhex(sel_hex)
        attendu = bytes.fromhex(digest_hex)
        calcule = hashlib.pbkdf2_hmac(
            "sha256", mot_de_passe.encode("utf-8"), sel, int(iterations)
        )
        return hmac.compare_digest(calcule, attendu)
    except (ValueError, TypeError):
        return False


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)

# Partie création et lecteur du token d'authentification.
# Le token est signé HMAC-SHA256 avec un secret côté serveur, et contient l'id et le nom d'utilisateur, ainsi qu'une date d'expiration. 
# Il est encodé en base64 pour être transporté dans l'en-tête Authorization.

def creer_token(user_id: int, username: str) -> dict:
    expiration = int(time.time()) + DUREE_TOKEN_SECONDES
    payload = {"uid": int(user_id), "sub": username, "exp": expiration}
    corps = _b64(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(AUTH_SECRET.encode("utf-8"), corps.encode("ascii"), hashlib.sha256).hexdigest()
    return {
        "access_token": f"{corps}.{signature}",
        "token_type": "bearer",
        "expires_in": DUREE_TOKEN_SECONDES,
        "expires_at": expiration,
        "id": int(user_id),
        "username": username,
    }


def lire_token(token: str) -> dict:
    try:
        corps, signature = token.split(".", 1)
    except ValueError as exc:
        raise ValueError("Jeton mal formé.") from exc
    attendue = hmac.new(AUTH_SECRET.encode("utf-8"), corps.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(attendue, signature):
        raise ValueError("Jeton invalide.")
    payload = json.loads(_b64_decode(corps))
    if int(payload.get("exp", 0)) < int(time.time()):
        raise ValueError("Jeton expiré.")
    if "uid" not in payload or "sub" not in payload:
        raise ValueError("Jeton incomplet.")
    return {"id": int(payload["uid"]), "username": str(payload["sub"]), "exp": int(payload["exp"])}
