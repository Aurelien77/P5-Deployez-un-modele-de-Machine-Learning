P5 — Déployez un modèle de Machine Learning
Application de prédiction RH : une API FastAPI charge un modèle
`LogisticRegression`, calcule une probabilité de départ, et une interface
web permet de tester le modèle puis d'enregistrer le résultat dans
PostgreSQL. L'interface n'est accessible qu'avec un compte : le mot de
passe est haché, la session dure 12 h, et chaque prédiction appartient à
l'utilisateur connecté.

La dernière version du logiciel est hebergé sur un serveur Hetzner à l'adresse suivante : 

https://predictions.1dream.art



```
Voici la composition des dossiers locaux du projet
```
| Dossier     | Rôle                                       |
| ----------- | ------------------------------------------ |
| `backend/`  | API, modèle ML, base de données            |
| `frontend/` | Interface HTML / CSS / JS servie par Nginx |
| `devops/`   | Docker Compose prod + playbooks Ansible    |
| `tests/`    | Tests pytest de l'API                      |
| `Data/`     | CSV d'initialisation de la base            |
```
---
Lancer le projet en local
Prérequis : Docker d'installer sur la machine local.
Les mots de passe ne sont pas dans `docker-compose.yml`.
Vous pouvez créer un fichier .env identique au fichier .env.example avec vos propres identifiants.


```bash
cp .env.example .env
```
Édite `.env` si besoin (Dev uniquement) :
```env
DB_USER=postgres
DB_PASSWORD=mysecretpassword
DB_NAME=rh_predictions_db
AUTH_SECRET=change-me
```
Pour démmarer l'application contenierisées la commande suivante s'effectue depuis la racine du projet :  

```bash
docker compose up --build
```
Frontend : http://localhost:8080
API / Swagger : http://localhost:8000/docs



Comptes et prédictions
L'interface reste masquée tant que l'utilisateur n'est pas connecté.
`POST /auth/register` crée le compte et renvoie un jeton de 12 h.
`POST /auth/login` ouvre une session. Le mot de passe est vérifié contre
une empreinte PBKDF2-SHA256 (200 000 itérations, sel aléatoire).
`GET /auth/moi` renvoie le compte du jeton.
`POST /auth/mot-de-passe` remplace l'empreinte après vérification de
l'ancien mot de passe.
`DELETE /auth/compte` supprime le compte et ses prédictions.
Le jeton est signé HMAC avec `AUTH_SECRET`. Il voyage dans
`Authorization: Bearer ...`, stocké côté navigateur dans `localStorage`.
Les routes protégées passent par `utilisateur_courant` : sans jeton valide,
la réponse est `401`.
`resultats.id_user` relie chaque prédiction à `users.id`
(`ON DELETE CASCADE`). `GET /resultats` ne renvoie que les lignes du compte
connecté. L'`id` employé n'est pas une colonne du modèle : il identifie la
fiche, le compte vient uniquement du jeton.
Les routes existent sous `/auth/...` et `/api/auth/...`. Le navigateur
appelle `/api/...` sur le port 8080. Nginx retire ce préfixe en transmettant
à FastAPI, qui reçoit `/auth/...`. La copie `/api` sert un appel direct sur
le port 8000. Les deux exécutent la même fonction.

Base de données
`users` et `resultats` sont créées au démarrage de l'API par
`database.py` (`Base.metadata.create_all`), à partir des modèles
`UserDB` et `ResultatDB`. Le contrôleur passe par l'ORM :
`db.query`, `db.add`, `db.commit`. La réponse n'est pas le modèle :
un dictionnaire, que FastAPI sérialise en JSON.
`employes_features` et `employes_cible` n'ont pas de classe modèle.
`db-init-csv` les crée depuis les CSV et les remplit en SQL brut.
`GET /annuaire` et `GET /employes/{id}` lisent ces tables avec `text()`.
Le filtre « a quitté » est appliqué dans `scripts.js`, pas dans le SQL.
Volumes :
`pgdata` conserve les fichiers Postgres après un `docker compose down`.
`./Data:/app/Data:ro` donne les CSV au script d'import, le temps de son
exécution.
`./backend/modeles:/app/modeles` permet de changer le `.pkl` sans
reconstruire l'image API.

Golden path (backend)
Chemin recommandé : une branche, un pipeline, des interrupteurs.
Pipeline : `.github/workflows/ci-cd.yml`
Interrupteurs : `golden-path.env` à la racine
```bash
ENABLE_STAGING_TESTS=true
ENABLE_PRODUCTION_RELEASE=true
ENABLE_SEMANTIC_RELEASE=true
```
Le premier job CI lit ce fichier et active ou saute les jobs.
Chemin 1 — `Stagging`
```bash
git push origin Stagging
```
Tests pytest + couverture
Si `ENABLE_SEMANTIC_RELEASE=true` : tag `vX.Y.Z-rc.N` + artefact
Actions `release-version`
Build images et Deploy restent ignorés sur `Stagging`.
Chemin 2 — `Main` (validation manuelle)
```bash
git checkout Main
git merge Stagging
git push origin Main
```
Build images récupère l'artefact `release-version` du dernier run
`Stagging` et tague `api:X.Y.Z` / `frontend:X.Y.Z`
Release production attend l'approbation de l'environnement
GitHub `production`, puis Ansible déploie
Secrets GitHub : `SSH_PRIVATE_KEY`, `PROD_HOST`, `ANSIBLE_VAULT_PASSWORD`.
```
feature/* → Stagging (tests + version) → Main (build via artefact) → Review → Prod
```

Rôle de chaque développeur
Développeur backend
Responsable de tout ce qui tourne derrière `/api` :
endpoints FastAPI (`backend/routes/`, `backend/controllers/`)
schéma Pydantic et modèle ORM (`backend/schemas/`, `backend/models/`)
comptes, hachage et jetons (`backend/auth_utils.py`)
chargement du `.pkl` et encodage des features
connexion PostgreSQL (`backend/database.py`)
import CSV (`script/insert_X_and_y_tables.py`)
tests dans `tests/test_api.py`
`backend/requirements.txt` et `backend/Dockerfile`
gardien du golden path : `Stagging` vert avant fusion vers `Main`
Documentation : backend/README.md
Développeur frontend
Responsable de l'interface :
`frontend/index.html`, `frontend/scripts.js`, `frontend/style.css`
`frontend/nginx.conf` (proxy `/api/` → backend)
`frontend/dockerfile`
`index.html` décrit la page. `scripts.js` pose les écouteurs et envoie les
`fetch`. Il consomme `POST /auth/login`, `POST /auth/register`,
`GET /colonnes`, `POST /predict`, `GET /resultats`, `GET /annuaire` et
`GET /employes/{id}`. Il ne touche ni au modèle ML ni au Vault.
Documentation : frontend/README.md

Architecture rapide
```
Navigateur
    │  :8080
    ▼
Nginx (frontend) ── /api/* ──► FastAPI (backend :8000)
                                    │
                                    ├── modeles/LogisticRegression.pkl
                                    └── PostgreSQL
                                         ├── users, resultats     (ORM)
                                         └── employes_*           (SQL brut)
```
Production : images GHCR + `devops/docker-compose.prod.yml` déployé par
Ansible (`devops/ansible/playbook_update.yml`).