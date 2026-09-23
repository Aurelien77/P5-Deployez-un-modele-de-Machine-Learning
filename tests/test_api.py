import pytest
from fastapi.testclient import TestClient
# Importez votre application FastAPI (adaptez le chemin d'import selon votre structure)
from Fast_API.app import app, COLONNES_MODELE

client = TestClient(app)

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
    response = client.post("/predict", json=payload)
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
    response = client.post("/predict", json=payload)
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
    
    # Sauvegarde
    res_save = client.post("/sauvegarder", json=payload_sauvegarde)
    assert res_save.status_code == 200
    data_save = res_save.json()
    assert "id" in data_save
    
    # Relecture de l'historique
    res_list = client.get("/resultats?limit=5")
    assert res_list.status_code == 200
    data_list = res_list.json()
    assert data_list["nb"] > 0
    
    # Vérification que notre entrée s'y trouve bien
    derniers = data_list["resultats"]
    trouve = any(item["prenom"] == "Alice" and item["nom"] == "Test" for item in derniers)
    assert trouve is True