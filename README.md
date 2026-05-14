# Credit Scoring — MLOps End-to-End

Implémentation et déploiement d'un modèle de scoring de défaut de paiement
de carte de crédit, dans une démarche MLOps complète.

## Dataset

**UCI Credit Card Default (Taiwan)** — 30 000 clients, 25 variables.

- Cible : `default.payment.next.month` (1 = défaut le mois suivant)
- Source : https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
- Fichier attendu : `data/raw/UCI_Credit_Card.csv`

## Stack technique

| Composant | Choix |
|-----------|-------|
| Tracking | **MLflow 2.17** + **PostgreSQL 16** (backend store) |
| Modèles | LogisticRegression, RandomForest, XGBoost |
| Déséquilibre | SMOTE (imbalanced-learn) |
| Explicabilité | SHAP |
| API | **FastAPI** + Uvicorn |
| UI | **Streamlit** |
| Drift | **Evidently AI** |
| CI/CD | GitHub Actions |
| Déploiement | Render.com (Docker) |

## Structure du projet

```
.
├── docker-compose.yml          # Stack locale (MLflow + Postgres)
├── .env.example                # Template variables d'environnement
├── requirements.txt
├── infra/mlflow/Dockerfile     # Image MLflow + driver psycopg2
├── data/
│   ├── raw/                    # CSV original (gitignored)
│   └── processed/              # Données nettoyées (gitignored)
├── notebooks/                  # EDA, expérimentation
├── src/scoring/                # Module Python du projet
├── api/                        # Application FastAPI
├── streamlit_app/              # Interface Streamlit
├── tests/                      # Tests unitaires et d'intégration
└── docs/                       # Spec, slides soutenance
```

## Prérequis

- Docker & Docker Compose v2
- Python 3.11+
- ~2 GB d'espace disque

## Démarrage rapide

```bash
# 1. Cloner et préparer l'environnement
cp .env.example .env

# 2. Télécharger le dataset depuis Kaggle
#    (placer UCI_Credit_Card.csv dans data/raw/)

# 3. Lancer l'infrastructure MLflow + Postgres
docker compose up -d

# 4. Vérifier que l'UI MLflow répond
open http://localhost:5050

# 5. Installer les dépendances Python en local
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Endpoints locaux

| Service | URL |
|---------|-----|
| MLflow UI | http://localhost:5050 |
| FastAPI (à venir) | http://localhost:8000/docs |
| Streamlit (à venir) | http://localhost:8501 |

## Étapes du projet (cf. énoncé)

1. ✅ Infrastructure MLflow + PostgreSQL
2. ⏳ Préparation et feature engineering des données
3. ⏳ Définition du score métier
4. ⏳ Entraînement et comparaison des modèles
5. ⏳ API REST + CI/CD + déploiement cloud
6. ⏳ Interface Streamlit
7. ⏳ Analyse data drift + soutenance

## Auteur

Examen MLOps — Master 2, 2026.
