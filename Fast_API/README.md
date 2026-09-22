# API de prédiction — démarrage rapide

## 1. Préparer le dossier

```
api_modele/
├── app.py
├── requirements.txt
└── modele.pkl        
                          
```

Renommez votre fichier téléchargé :
```bash
mv modele_top1_ElasticNet_LogReg_Recall_test.pkl modele.pkl
```

## 2. Installer les dépendances (dans un environnement virtuel dédié)

```bash
python -m venv venv
source venv/bin/activate      # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

⚠️ La version de `scikit-learn` dans `requirements.txt` doit être
**identique** à celle utilisée pour entraîner le modèle dans votre
notebook. Vérifiez avec :
```python
import sklearn; print(sklearn.__version__)
```
dans le notebook, et ajustez `requirements.txt` si besoin.

## 3. Lancer l'API

```bash
uvicorn app:app --reload --port 8000
```

## 4. Tester

Documentation interactive (Swagger) :
http://127.0.0.1:8000/docs

Lister les colonnes attendues :
```bash
curl http://127.0.0.1:8000/colonnes
```

Faire une prédiction :
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "features": {
          "age": 34,
          "revenu_mensuel": 3200,
         ...
        },
        "seuil": 0.5
      }'
```

Réponse :
```json
{
  "prediction": 1,
  "probabilite": 0.62,
  "seuil_utilise": 0.5
}
```

## 5. Déploiement


