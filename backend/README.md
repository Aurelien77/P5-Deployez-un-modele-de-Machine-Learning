# API de prédiction RH — backend structuré

L'application FastAPI est organisée en **routes** (HTTP) et **controllers** (logique métier).

```
backend/
├── app.py                    # Point d'entrée mince (FastAPI + middlewares)
├── database.py               # Engine SQLAlchemy, session, init Postgres
├── models/
│   └── resultat.py           # Modèle ORM `resultats`
├── schemas/
│   └── prediction.py         # Contrats Pydantic (requêtes)
├── controllers/
│   ├── modele_controller.py  # Chargement ML, encodage, prédiction
│   ├── resultat_controller.py# Sauvegarde et listing en base
│   └── root_controller.py    # Accueil / index.html
├── routes/
│   ├── root.py               # GET /
│   ├── prediction.py         # GET /colonnes, POST /predict
│   ├── resultats.py          # POST /sauvegarder, GET /resultats
│   └── debug.py              # GET /debug/modele/{modele}
├── modeles/
│   └── LogisticRegression.pkl
├── script/
│   └── init_db_from_csv.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Lancer en local

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Swagger : http://127.0.0.1:8000/docs

## Tests

Depuis la racine du dépôt (avec `PYTHONPATH=.`) :

```bash
pytest tests/test_api.py
```

Les tests continuent d'importer `from backend.app import app, COLONNES_MODELE`.
