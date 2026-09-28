# Frontend — interface de prédiction RH

Interface statique (HTML / CSS / JavaScript) servie par **Nginx**.
Elle permet de renseigner une fiche employé, d'appeler l'API de
prédiction et d'afficher / sauvegarder le résultat.

Aucun framework JS : un seul fichier `index.html` contient le markup et
le script.

## Structure

```
frontend/
├── index.html      # UI + appels fetch vers l'API
├── style.css       # Mise en page
├── nginx.conf      # Fichiers statiques + proxy /api/ → backend
└── dockerfile      # Image nginx:alpine
```

## Comportement

`API_BASE` est fixé à `"/api"` dans `index.html`. Nginx réécrit :

```
/api/colonnes     →  http://api:8000/colonnes
/api/predict      →  http://api:8000/predict
/api/sauvegarder  →  http://api:8000/sauvegarder
/api/resultats    →  http://api:8000/resultats
```

L'utilisateur :

1. Saisit prénom / nom, choisit le modèle (`top1` = LogisticRegression)
   et le seuil (défaut `0.37`).
2. Complète les variables sources (dont celles qui calculent
   `formations_par_an` et `poste_x_niveau`).
3. Clique pour obtenir la probabilité et la classe (0 / 1).
4. Peut sauvegarder la fiche en base et relire les derniers résultats.

## Lancer en local

### Avec tout le stack (recommandé)

Depuis la racine du dépôt :

```bash
docker compose up --build
```

Ouvrir http://localhost:8080

Le service `frontend` dépend de `api`. Nginx écoute sur le port 80 du
conteneur, publié en 8080 sur l'hôte.

### Image frontend seule

Utile uniquement si l'API tourne déjà ailleurs. Adapter alors le
`proxy_pass` de `nginx.conf`.

```bash
cd frontend
docker build -t p5-frontend -f dockerfile .
docker run --rm -p 8080:80 p5-frontend
```

Sans le service `api` sur le réseau Docker, les appels `/api/*` échouent.

## Rôle du développeur frontend

- Faire évoluer l'UI (`index.html`, `style.css`) sans casser les noms de
  champs attendus par `POST /predict` et `POST /sauvegarder`.
- Si un nouveau champ modèle apparaît côté API, l'ajouter au formulaire
  **et** vérifier `GET /colonnes`.
- Ne pas embarquer de secret ni de logique métier (seuil métier, encodage
  des catégories) : ça reste dans le backend.
- Vérifier le proxy Nginx (`location /api/`) après tout changement de
  chemin d'API.
- Livraison : même golden path que le backend (le job `Main` construit
  aussi l'image `frontend` et la pousse sur GHCR). Une PR front passe
  donc par `Stagging` (les tests API restent le filet de sécurité du
  contrat) puis `Main` + validation manuelle pour la prod.

Contrats à respecter (ne pas les changer sans accord backend) :

| Appel | Usage dans l'UI |
|-------|-----------------|
| `GET /colonnes?modele=` | Construire / valider le formulaire |
| `POST /predict` | Afficher probabilité + prédiction |
| `POST /sauvegarder` | Persister la fiche |
| `GET /resultats?limit=` | Historique |

## Production

L'image est publiée par la CI (`ghcr.io/<org>/<repo>/frontend:<sha>`).
`devops/docker-compose.prod.yml` la branche sur le réseau `edge-network`
(reverse-proxy hôte). Le `nginx.conf` continue de proxifier `/api/` vers
le service Docker `api`.
