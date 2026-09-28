# Backend — API de prédiction RH

API **FastAPI** qui charge un modèle scikit-learn (`LogisticRegression.pkl`),
expose une prédiction de risque RH et persiste les résultats dans
PostgreSQL.

## Structure

```
backend/
├── app.py                     # Point d'entrée (FastAPI + CORS + init)
├── database.py                # Engine SQLAlchemy, session, init Postgres
├── models/
│   └── resultat.py            # Table ORM `resultats`
├── schemas/
│   └── prediction.py          # Contrats Pydantic (predict / sauvegarde)
├── controllers/
│   ├── modele_controller.py   # Chargement ML, encodage, prédiction
│   ├── resultat_controller.py # Insert + listing en base
│   └── root_controller.py     # Accueil
├── routes/
│   ├── root.py                # GET /
│   ├── prediction.py          # GET /colonnes, POST /predict
│   ├── resultats.py           # POST /sauvegarder, GET /resultats
│   └── debug.py               # GET /debug/modele/{modele}
├── modeles/
│   └── LogisticRegression.pkl
├── script/
│   └── init_db_from_csv.py    # Import Data/X.csv + Data/y.csv
├── requirements.txt
└── Dockerfile
```

Les routes restent minces : elles délèguent toute la logique aux
controllers.

## Endpoints

| Méthode | Chemin | Rôle |
|---------|--------|------|
| GET | `/` | Page d'accueil (si HTML présent) |
| GET | `/docs` | Swagger |
| GET | `/colonnes?modele=top1` | Liste des features attendues |
| POST | `/predict` | Probabilité + classe selon le seuil |
| POST | `/sauvegarder` | Enregistre une prédiction en base |
| GET | `/resultats?limit=20` | Derniers résultats |
| GET | `/debug/modele/top1` | Infos techniques du modèle chargé |

Alias du modèle dans le code : `top1` → `LogisticRegression.pkl`.

## Lancer sans Docker

PostgreSQL doit tourner (voir `docker-compose.yml` à la racine, service
`postgres` uniquement).

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export DB_HOST=localhost
export DB_PORT=5432
export DB_USER=postgres
export DB_PASSWORD=mysecretpassword
export DB_NAME=rh_predictions_db

uvicorn app:app --reload --port 8000
```

Swagger : http://127.0.0.1:8000/docs

Variables d'environnement lues par `database.py` :

| Variable | Défaut local |
|----------|----------------|
| `DB_HOST` | `localhost` |
| `DB_PORT` | `5432` |
| `DB_USER` | `postgres` |
| `DB_PASSWORD` | `mysecretpassword` |
| `DB_NAME` | `rh_predictions_db` |

## Lancer avec Docker

Depuis la **racine** du dépôt :

```bash
docker compose up --build api postgres db-init-csv
```

L'image backend démarre `uvicorn app:app --host 0.0.0.0 --port 8000`.

## Tests

Depuis la racine du dépôt (`PYTHONPATH=.`) :

```bash
pip install -r backend/requirements.txt pytest pytest-cov httpx
PYTHONPATH=. pytest tests/test_api.py --cov=backend --cov-report=term-missing
```

Les tests importent `from backend.app import app, COLONNES_MODELE`.
Ils ont besoin d'une base PostgreSQL joignable (Compose local ou service
CI).

## Rôle du développeur backend

- Faire évoluer les routes / controllers / schémas sans casser le contrat
  consommé par le frontend (`/colonnes`, `/predict`, `/sauvegarder`,
  `/resultats`).
- Garder `scikit-learn==1.9.0` aligné sur la version d'entraînement du
  `.pkl` (sinon le chargement échoue).
- Ajouter un test dans `tests/test_api.py` pour chaque nouveau
  comportement.
- Suivre le **golden path** décrit à la racine :
  - développer sur une branche feature
  - pousser / fusionner sur `Stagging` → la CI lance les tests
  - seulement après tests verts, fusionner sur `Main` → release après
    **validation manuelle** de l'environnement GitHub `production`

### Activer / couper le golden path

Dans [`.github/workflows/ci-cd.yml`](../.github/workflows/ci-cd.yml) :

- `ENABLE_STAGING_TESTS` — job tests sur `Stagging`
- `ENABLE_PRODUCTION_RELEASE` — build images + deploy sur `Main`

Un nouveau développeur peut aussi aller dans **Actions → Run workflow**
et cocher `activer_tests_staging` ou `activer_release_production`.

## Dépendances notables

Voir `requirements.txt` : FastAPI, Uvicorn, pandas, numpy,
scikit-learn 1.9.0, SQLAlchemy, psycopg, joblib, httpx.
