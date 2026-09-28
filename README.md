# P5 — Déployez un modèle de Machine Learning

Application de **prédiction RH** : une API FastAPI charge un modèle
`LogisticRegression`, calcule une probabilité (départ / risque), et une
interface web permet de tester le modèle puis de sauvegarder le résultat
en PostgreSQL.

| Dossier | Rôle |
|---------|------|
| `backend/` | API, modèle ML, base de données |
| `frontend/` | Interface HTML / CSS / JS servie par Nginx |
| `devops/` | Docker Compose prod + playbooks Ansible |
| `tests/` | Tests pytest de l'API |
| `Data/` | CSV d'initialisation de la base |

---

## Lancer le projet en local

Prérequis : Docker et Docker Compose.

Les mots de passe **ne sont plus dans** `docker-compose.yml`.
Compose lit un fichier **`.env`** à la racine (ignoré par git).

```bash
cp .env.example .env
```

Édite `.env` si besoin (Dev uniquement) :

```env
DB_USER=postgres
DB_PASSWORD=mysecretpassword
DB_NAME=rh_predictions_db
```

Puis :

```bash
docker compose up --build
```

- Frontend : http://localhost:8080
- API / Swagger : http://localhost:8000/docs
- PostgreSQL : `localhost:5432` (valeurs de ton `.env`)

Le service `db-init-csv` charge `Data/X.csv` et `Data/y.csv` au premier
démarrage.

### Fichiers d'environnement

| Fichier | Dans git ? | Rôle |
|---------|------------|------|
| `.env.example` | oui | modèle sans secret, à copier |
| `.env` | **non** | secrets Dev local |
| `docker-compose.yml` | oui | stack Dev, lit `${DB_PASSWORD}` |
| `devops/docker-compose.prod.yml` | oui | stack prod, lit `${DB_PASSWORD}` |
| `devops/.env.prod` | **non** | généré sur le serveur par Ansible |
| `devops/ansible/vault.yml` | oui (chiffré) | secret prod `vault_db_password` |

Ne commite jamais `.env` ni `devops/.env.prod`. Voir `.gitignore`.

La prod n'utilise pas le compose racine. Ansible déchiffre le Vault
(`ANSIBLE_VAULT_PASSWORD` côté GitHub), écrit `devops/.env.prod`, puis
lance `docker-compose.prod.yml`.

---

## Golden path (backend)

Chemin recommandé : une branche, un pipeline, des interrupteurs.

- Pipeline : [`.github/workflows/ci-cd.yml`](.github/workflows/ci-cd.yml)
- Interrupteurs : [`golden-path.env`](golden-path.env) **à la racine**

```bash
ENABLE_STAGING_TESTS=true
ENABLE_PRODUCTION_RELEASE=true
ENABLE_SEMANTIC_RELEASE=true
```

Le premier job CI lit ce fichier et active ou saute les jobs.

### Chemin 1 — `Stagging`

```bash
git push origin Stagging
```

1. Tests pytest + couverture
2. Si `ENABLE_SEMANTIC_RELEASE=true` : tag `vX.Y.Z-rc.N` + artefact
   Actions `release-version`

Build images et Deploy restent **ignorés** sur `Stagging`.

### Chemin 2 — `Main` (validation manuelle)

```bash
git checkout Main
git merge Stagging
git push origin Main
```

1. **Build images** récupère l'artefact `release-version` du dernier run
   `Stagging` et tague `api:X.Y.Z` / `frontend:X.Y.Z`
2. **Release production** attend l'approbation de l'environnement
   GitHub `production`, puis Ansible déploie

Secrets GitHub : `SSH_PRIVATE_KEY`, `PROD_HOST`, `ANSIBLE_VAULT_PASSWORD`.

```
feature/* → Stagging (tests + version) → Main (build via artefact) → Review → Prod
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
- gardien du golden path : `Stagging` vert avant fusion vers `Main`

Documentation : [backend/README.md](backend/README.md)

### Développeur frontend

Responsable de l'interface :

- `frontend/index.html`, `frontend/style.css`
- `frontend/nginx.conf` (proxy `/api/` → backend)
- `frontend/dockerfile`

Il consomme `GET /colonnes`, `POST /predict`, `POST /sauvegarder`,
`GET /resultats`. Il ne touche ni au modèle ML ni au Vault.

Documentation : [frontend/README.md](frontend/README.md)

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
