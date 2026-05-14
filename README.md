# Credit Scoring — MLOps End-to-End

Implémentation et déploiement d'un modèle de scoring de défaut de paiement
de carte de crédit, dans une démarche MLOps complète.

## 🎯 Quick links

- **Repo GitHub** : https://github.com/oumar390/examen-mlops
- **MLflow UI** (local) : http://localhost:5050
- **FastAPI** (local) : http://localhost:8000 · Swagger : http://localhost:8000/docs
- **Streamlit** (local) : http://localhost:8501
- **Slides** : [`docs/SOUTENANCE.md`](docs/SOUTENANCE.md)
- **Drift report** : [`docs/drift_report.html`](docs/drift_report.html)

## Dataset

**UCI Credit Card Default (Taiwan)** — 30 000 clients, 25 variables.

- Cible : `default.payment.next.month` (1 = défaut le mois suivant)
- Source : https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
- Fichier attendu : `data/raw/UCI_Credit_Card.csv`

## Stack technique

| Composant | Choix |
|-----------|-------|
| Tracking | **MLflow 2.17** + **PostgreSQL 16** (backend store) |
| Modèles | LogisticRegression, RandomForest, **XGBoost** (winner) |
| Déséquilibre | SMOTE (in pipeline, anti-leakage) |
| Score métier | `business_gain` custom (ratio FN:FP = 5:1, Bâle II/III) |
| Explicabilité | **SHAP** TreeExplainer |
| API | **FastAPI** + Uvicorn + Pydantic |
| UI | **Streamlit** + Plotly |
| Drift | **Evidently AI** |
| CI | GitHub Actions (lint + tests + docker build) |
| Déploiement | **Render.com** (Docker, blueprint `render.yaml`) |

## Structure du projet

```
.
├── docker-compose.yml          # Stack locale complète (mlflow + postgres + api + streamlit)
├── render.yaml                 # Render Blueprint pour le déploiement
├── requirements.txt            # Deps Python du projet
├── .env.example                # Variables d'environnement template
├── .github/workflows/ci.yml    # CI GitHub Actions
├── infra/mlflow/Dockerfile     # Image MLflow + psycopg2
├── data/
│   ├── raw/                    # CSV original (gitignored)
│   └── processed/              # Données nettoyées (parquet)
├── notebooks/
│   ├── 01_eda.ipynb            # EDA + cleaning + FE inline (109 cells)
│   ├── 02_business_score.ipynb # Justif + visu du score métier (21 cells)
│   ├── 03_training.ipynb       # Pipeline + GridSearch + SHAP (32 cells)
│   └── 04_drift.ipynb          # Monitoring drift (13 cells)
├── src/scoring/                # Module Python core
│   ├── data.py                 # load_raw / clean / engineer_features
│   ├── business_score.py       # cost / gain / scorer / threshold opt
│   ├── preprocessing.py        # ColumnTransformer factory
│   ├── train.py                # CLI training pipeline (MLflow logging)
│   └── drift.py                # Evidently report generator
├── api/                        # FastAPI service
│   ├── main.py                 # 5 endpoints
│   ├── schemas.py              # Pydantic models
│   ├── Dockerfile
│   └── requirements.txt        # Slim runtime deps
├── streamlit_app/              # UI Streamlit
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── tests/                      # 31 tests pytest (unit + integration)
├── models/                     # Production artefacts
│   ├── best_model.joblib       # XGBoost pipeline complet
│   └── metadata.json
└── docs/
    ├── SOUTENANCE.md           # Slides Marp
    ├── drift_report.html       # Rapport Evidently
    └── figures/                # ~30 plots
```

## 🚀 Démarrage rapide

```bash
# 1. Cloner et préparer l'environnement
git clone https://github.com/oumar390/examen-mlops.git
cd examen-mlops
cp .env.example .env

# 2. Télécharger le dataset depuis Kaggle dans data/raw/UCI_Credit_Card.csv

# 3. Créer le venv Python 3.11.9
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Lancer toute la stack en Docker
docker compose up -d --build

# 5. Entraîner le modèle (loggué dans MLflow + sauvegardé dans models/)
python -m scoring.train

# 6. Tester l'API en local
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @docs/sample_request.json

# 7. Ouvrir l'UI
open http://localhost:8501
```

## 🧪 Tests

```bash
pytest tests/ -v             # 31 tests
```

## ☁️ Déploiement Render

Le repo contient un `render.yaml` (Blueprint). Pour déployer :

1. Aller sur https://render.com → **New** → **Blueprint**
2. Connecter le repo `oumar390/examen-mlops`
3. Render détecte `render.yaml` et crée le service `credit-scoring-api`
4. Premier déploiement → ~5-10 min de build
5. URL fournie : `https://credit-scoring-api.onrender.com`

Auto-deploy activé : chaque push sur `main` redéploie automatiquement.

## 📋 Étapes de l'examen (couvertes)

| Étape | Statut | Livrables |
|-------|--------|-----------|
| 1. Environnement MLFlow | ✅ | `docker-compose.yml`, Postgres backend, UI live |
| 2. Préparation données | ✅ | `01_eda.ipynb`, `src/scoring/data.py`, dataset processed |
| 3. Score métier | ✅ | `02_business_score.ipynb`, `business_score.py`, 15 tests |
| 4. Entraînement & comparaison | ✅ | `03_training.ipynb`, `train.py`, 3 runs MLflow, SHAP |
| 5. API REST + CI/CD | ✅ | FastAPI, Docker, `ci.yml`, `render.yaml` |
| 6. Streamlit | ✅ | `streamlit_app/app.py`, gauge interactif |
| 7. Data drift + soutenance | ✅ | `04_drift.ipynb`, Evidently report, `SOUTENANCE.md` |

## 🔑 Résultats

| Modèle | Test business_gain | Recall | AUC |
|--------|---------------------|--------|-----|
| LogReg | 0.461 | 65 % | 0.74 |
| RandomForest | 0.480 | 72 % | 0.78 |
| **XGBoost** 🏆 | **0.484** | **76 %** | 0.77 |

**Modèle retenu** : XGBoost (n_estimators=200, max_depth=3, lr=0.1)
**Seuil de décision optimisé** : 0.28

## Auteur

Examen MLOps — Master 2, 2026.
