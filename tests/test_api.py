import pytest
from fastapi.testclient import TestClient
# Importez votre application FastAPI (adaptez le chemin d'import selon votre structure)
from backend.app import app, COLONNES_MODELE

client = TestClient(app)


def _auth_headers(suffix: str = "api"):
    """Crée un compte jetable et renvoie le header Bearer (jeton 12 h)."""
    import uuid
    username = f"{suffix}_{uuid.uuid4().hex[:10]}"
    password = "secret123"
    reg = client.post("/auth/register", json={"username": username, "password": password})
    assert reg.status_code == 201, reg.text
    token = reg.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_read_root():
    """Test de l'endpoint racine (charge l'interface HTML ou renvoie 404 si absente)."""
    response = client.get("/")
    # Selon la présence ou non de index.html, on s'attend soit à un 200, soit à un 404 géré
    assert response.status_code in [200, 404]

def test_get_colonnes_valide():
    """Vérifie la récupération des colonnes pour un modèle existant."""
    response = client.get("/colonnes?modele=top1")
    assert response.status_code == 200
    data = response.json()
    assert "colonnes" in data
    assert isinstance(data["colonnes"], list)
    assert len(data["colonnes"]) > 0

def test_get_colonnes_modele_inconnu():
    """Vérifie le comportement avec un modèle non configuré."""
    response = client.get("/colonnes?modele=modele_inexistant")
    assert response.status_code == 200
    data = response.json()
    # Doit retomber sur les colonnes par défaut
    assert data["source"] == "defaut_modele_non_charge"

def test_debug_modele():
    """Vérifie l'endpoint de debug d'un modèle."""
    response = client.get("/debug/modele/top1")
    assert response.status_code == 200
    data = response.json()
    assert "type_model" in data
    assert "a_scaler" in data

def test_predict_nominal():
    """Test fonctionnel d'une prédiction valide avec des données réalistes."""
    payload = {
        "modele": "top1",
        "seuil": 0.37,
        "features": {
            "age": 35,
            "revenu_mensuel": 5000,
            "annees_dans_l_entreprise": 5,
            "distance_domicile_travail": 10,
            "annes_sous_responsable_actuel": 2,
            "satisfaction_employee_environnement": 4,
            "niveau_hierarchique_poste": 2,
            "satisfaction_min": 2,
            "nb_formations_suivies": 2,
            "annee_experience_totale": 10,
            "niveau_education": 3,
            "statut_marital": "Célibataire",
            "frequence_deplacement": "Occasionnel",
            "heure_supplementaires": "Non"
        }
    }
    response = client.post("/predict", json=payload, headers=_auth_headers("pred"))
    assert response.status_code == 200
    data = response.json()
    
    assert data["modele_utilise"] == "top1"
    assert "probabilite" in data
    assert 0.0 <= data["probabilite"] <= 1.0
    assert data["prediction"] in [0, 1]

def test_predict_modele_invalide():
    """Test de cas limite : appel d'un modèle qui n'existe pas."""
    payload = {
        "modele": "modele_fantome",
        "features": {"age": 30}
    }
    response = client.post("/predict", json=payload, headers=_auth_headers("fantome"))
    assert response.status_code == 404

def test_sauvegarder_et_lister_resultats():
    """Test d'intégration : sauvegarde d'une prédiction en base et relecture via /resultats."""
    payload_sauvegarde = {
        "prenom": "Alice",
        "nom": "Test",
        "modele_utilise": "top1",
        "probabilite_de_quitter": 0.85,
        "prediction": 1,
        "libelle_prediction": "À risqué de partir",
        "seuil_applique": 0.37,
        "features": {"age": 28, "revenu_mensuel": 3000}
    }
    
    headers = _auth_headers("alice")
    # Sauvegarde
    res_save = client.post("/sauvegarder", json=payload_sauvegarde, headers=headers)
    assert res_save.status_code == 200
    data_save = res_save.json()
    assert "id" in data_save
    assert data_save["id_user"]
    
    # Relecture de l'historique
    res_list = client.get("/resultats?limit=5", headers=headers)
    assert res_list.status_code == 200
    data_list = res_list.json()
    assert data_list["nb"] > 0
    
    # Vérification que notre entrée s'y trouve bien
    derniers = data_list["resultats"]
    trouve = any(item["prenom"] == "Alice" and item["nom"] == "Test" for item in derniers)
    assert trouve is True


def test_detail_resultat_et_filtre_nom():
    """Couvre GET /resultats/{id} et les filtres prenom/nom."""
    payload = {
        "prenom": "Bruno",
        "nom": "Couverture",
        "modele_utilise": "top1",
        "probabilite_de_quitter": 0.42,
        "prediction": 1,
        "libelle_prediction": "A risque de partir",
        "seuil_applique": 0.37,
        "features": {"age": 44},
    }
    headers = _auth_headers("bruno")
    sauve = client.post("/sauvegarder", json=payload, headers=headers)
    assert sauve.status_code == 200
    identifiant = sauve.json()["id"]

    detail = client.get(f"/resultats/{identifiant}", headers=headers)
    assert detail.status_code == 200
    assert detail.json()["prenom"] == "Bruno"
    assert detail.json()["source_features"] == "details_json"

    filtre = client.get("/resultats", params={"prenom": "Bruno", "nom": "Couverture", "limit": 5}, headers=headers)
    assert filtre.status_code == 200
    assert any(item["id"] == identifiant for item in filtre.json()["resultats"])

    inconnu = client.get("/resultats/99999999", headers=headers)
    assert inconnu.status_code == 404


def test_annuaire_et_employe():
    """Couvre /annuaire et /employes/{id}, tables presentes ou non."""
    annuaire = client.get("/annuaire", params={"q": "1", "limit": 5})
    assert annuaire.status_code in (200, 500)
    if annuaire.status_code == 200:
        assert "personnes" in annuaire.json()

    employe = client.get("/employes/1")
    assert employe.status_code in (200, 404, 500)


def test_debug_modele_inconnu_et_accueil_fichier(tmp_path, monkeypatch):
    """Couvre le 404 de debug et le FileResponse de l'accueil."""
    debug = client.get("/debug/modele/modele_inexistant")
    assert debug.status_code == 404

    index = tmp_path / "index.html"
    index.write_text("<html>accueil</html>", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    accueil = client.get("/")
    assert accueil.status_code == 200


def test_auth_compte():
    """Création, connexion, rejet d'un mot de passe faux, suppression."""
    import uuid
    username = f"auth_{uuid.uuid4().hex[:10]}"
    password = "secret123"
    cree = client.post("/auth/register", json={"username": username, "password": password})
    assert cree.status_code == 201
    assert cree.json()["expires_in"] == 12 * 60 * 60
    token = cree.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    doublon = client.post("/auth/register", json={"username": username, "password": password})
    assert doublon.status_code == 409

    mauvais = client.post("/auth/login", json={"username": username, "password": "mauvais_mdp"})
    assert mauvais.status_code == 401

    moi = client.get("/auth/moi", headers=headers)
    assert moi.status_code == 200
    assert moi.json()["username"] == username

    sans = client.get("/resultats")
    assert sans.status_code == 401

    suppr = client.delete("/auth/compte", headers=headers)
    assert suppr.status_code == 200
    encore = client.get("/auth/moi", headers=headers)
    assert encore.status_code == 401
