# Examen MLOps - API de scoring binaire

Ce projet couvre le cycle demandé dans l'énoncé: préparation des données, score métier,
comparaison de modèles, suivi MLflow, API FastAPI, interface Streamlit, CI/CD et stratégie
de monitoring du drift.

## Choix métier

Le cas d'usage retenu est un scoring binaire de risque médical à partir du dataset
`breast_cancer` de scikit-learn. La classe positive vaut `1` pour un cas malin à haut risque.
Un faux négatif est pondéré 5 fois plus qu'un faux positif, car manquer un cas malin est
beaucoup plus coûteux qu'une alerte inutile.

## Installation

Python 3.11 est recommandé.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## MLflow

```bash
docker compose up -d mlflow
export MLFLOW_TRACKING_URI=http://localhost:5050
export ENABLE_MLFLOW=true
python -m scoring_app.train
```

Interface MLflow: <http://localhost:5050>

## API et interface

Après l'entraînement, le modèle est sauvegardé dans `models/best_model.joblib`.

```bash
uvicorn scoring_app.api:app --reload --host 0.0.0.0 --port 8000
streamlit run src/scoring_app/streamlit_app.py
```

Endpoints principaux:

- `GET /health`
- `GET /features`
- `GET /example`
- `POST /predict`

Exemple minimal avec le payload complet fourni par l'API:

```bash
curl http://localhost:8000/example
```

Copier ensuite l'objet `features` dans `POST /predict`.

## Drift monitoring

Le script `scoring_app.drift` calcule un Population Stability Index par variable et classe
les variables en `ok`, `watch` ou `alert`.

```bash
python -m scoring_app.drift data/current_batch.csv --output reports/drift_report.json
```

Stratégie proposée:

- surveiller le PSI des variables les plus importantes, le taux de prédictions haut risque,
  la distribution des probabilités et les métriques métier sur les labels retardés;
- déclencher une alerte si PSI >= 0.25 sur une variable majeure ou si la performance métier
  baisse sous le seuil validé;
- réentraîner le modèle avec validation MLflow, revue des métriques et déploiement progressif.

## CI/CD

Le workflow `.github/workflows/ci.yml` exécute lint, tests avec couverture et build Docker.
Pour un déploiement cloud, connecter le job Docker à Render, Railway, AWS ECS ou GCP Cloud Run
avec des secrets GitHub Actions.

## Soutenance

Plan conseillé:

1. Contexte métier et coût FP/FN.
2. Données, nettoyage et features.
3. Score métier et baseline.
4. Comparaison Logistic Regression vs Random Forest avec MLflow.
5. Démo API FastAPI et Streamlit.
6. CI/CD, Docker et stratégie de monitoring drift.
