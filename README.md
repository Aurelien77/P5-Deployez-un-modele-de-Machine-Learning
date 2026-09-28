# P5 — Déployez un modèle de Machine Learning

Application de **prédiction RH** : une API FastAPI charge un modèle
`LogisticRegression`, calcule une probabilité (départ / risque), et une
interface web permet de tester le modèle puis de sauvegarder le résultat
en PostgreSQL.

Le dépôt est découpé en trois zones :

| Dossier | Rôle |
|---------|------|
| `backend/` | API, modèle ML, base de données |
| `frontend/` | Interface HTML / CSS / JS servie par Nginx |
| `devops/` | Docker Compose prod + playbooks Ansible |
| `tests/` | Tests pytest de l'API |
| `Data/` | CSV d'initialisation de la base |

## Lancer le projet en local

Prérequis : Docker et Docker Compose.

```bash
docker compose up --build
```

- Frontend : http://localhost:8080
- API / Swagger : http://localhost:8000/docs
- PostgreSQL : `localhost:5432` (`postgres` / `mysecretpassword` / `rh_predictions_db`)

Le service `db-init-csv` charge `Data/X.csv` et `Data/y.csv` au premier
démarrage.

---

## Golden path (backend)

Le **golden path** est le seul chemin recommandé pour un nouveau
développeur : une branche, un fichier CI, des variables à activer.

Fichier : [`.github/workflows/ci-cd.yml`](.github/workflows/ci-cd.yml)

### Variables à activer

Dans le workflow (bloc `env`) ou via
**Settings → Secrets and variables → Actions → Variables** :

| Variable | Défaut | Effet |
|----------|--------|--------|
| `ENABLE_STAGING_TESTS` | `true` | Autorise le job de tests |
| `ENABLE_PRODUCTION_RELEASE` | `true` | Autorise build + release |

On peut aussi lancer le workflow à la main
(**Actions → CI/CD Pipeline → Run workflow**) et cocher :

- `activer_tests_staging`
- `activer_release_production`

### Chemin 1 — tests sur `Stagging`

```bash
git checkout Stagging
git pull
git checkout -b feature/ma-tache
# ... commits ...
git checkout Stagging
git merge feature/ma-tache
git push origin Stagging
```

Le job **Tests (golden path Stagging)** démarre :

1. PostgreSQL de service
2. installation des dépendances
3. `pytest tests/test_api.py` + couverture
4. artefacts HTML / XML dans l'onglet Actions

Mettre `ENABLE_STAGING_TESTS=false` (variable dépôt) désactive ce chemin.

### Chemin 2 — release sur `Main` (validation manuelle)

Quand les tests `Stagging` sont verts :

```bash
git checkout Main
git merge Stagging
git push origin Main
```

1. Job **Build images** : construction et push
   `ghcr.io/<org>/<repo>/api` et `.../frontend` (tag SHA + `latest`).
2. Job **Release production** : attaché à l'environnement GitHub
   `production`. Tant qu'un reviewer n'a pas approuvé, le déploiement
   Ansible n'est **pas** lancé.

Configurer une fois :

1. GitHub → **Settings → Environments → New environment**
   - `staging` (sans reviewer)
   - `production` → activer **Required reviewers**
2. Secrets : `SSH_PRIVATE_KEY`, `PROD_HOST`, `ANSIBLE_VAULT_PASSWORD`

Mettre `ENABLE_PRODUCTION_RELEASE=false` désactive ce chemin.

```
feature/*  →  Stagging (tests auto)  →  Main (build)  →  Review  →  Prod
```

---

## Rôle de chaque développeur

### Développeur backend

Responsable de tout ce qui tourne derrière `/api` :

- endpoints FastAPI (`backend/routes/`, `backend/controllers/`)
- schéma Pydantic et modèle ORM (`backend/schemas/`, `backend/models/`)
- chargement du `.pkl` et encodage des features
- connexion PostgreSQL (`backend/database.py`)
- tests dans `tests/test_api.py`
- `backend/requirements.txt` et `backend/Dockerfile`
- **gardien du golden path CI** : un push `Stagging` doit rester vert
  avant toute fusion vers `Main`

Documentation détaillée : [backend/README.md](backend/README.md)

### Développeur frontend

Responsable de l'interface utilisateur :

- `frontend/index.html` (formulaire employé, appels `fetch`)
- `frontend/style.css`
- `frontend/nginx.conf` (fichiers statiques + proxy `/api/` → API)
- `frontend/dockerfile` (image Nginx)

Il ne modifie pas le modèle ML ni la base. Il consomme les contrats
`GET /colonnes`, `POST /predict`, `POST /sauvegarder`, `GET /resultats`.
En local via Compose, Nginx expose le front sur le port 8080 et relaie
`/api` vers le service `api`.

Documentation détaillée : [frontend/README.md](frontend/README.md)

---

## Architecture rapide

```
Navigateur
    │  :8080
    ▼
Nginx (frontend) ── /api/* ──► FastAPI (backend :8000)
                                      │
                                      ├── modeles/LogisticRegression.pkl
                                      └── PostgreSQL
```

Production : images GHCR + `devops/docker-compose.prod.yml` déployé par
Ansible (`devops/ansible/playbook_update.yml`).
