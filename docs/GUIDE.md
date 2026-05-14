# 📚 Guide complet du projet MLOps — pour la soutenance

> Document à lire avant la soutenance. Il explique **quoi**, **pourquoi**, et
> **comment** pour chaque décision. Il te servira aussi à répondre aux
> questions du jury.

---

## 0. Le pitch en 30 secondes

> "J'ai implémenté un système de credit scoring end-to-end pour prédire le
> défaut de paiement de cartes de crédit. Le projet couvre tout le cycle
> MLOps : tracking des expériences avec MLflow + Postgres, feature
> engineering business-driven, score métier custom asymétrique
> (FN coûte 5× FP), comparaison de 3 modèles via GridSearchCV avec
> SMOTE et SHAP, API FastAPI dockerisée, CI/CD GitHub Actions vers
> Render, interface Streamlit, et monitoring drift avec Evidently AI.
> Le modèle gagnant — XGBoost — atteint **76% de recall** avec un
> business gain de **0.484** sur le test set."

---

## 1. Le dataset — UCI Credit Card Default

### Contexte business

- **Source** : université Chung Hua, Taiwan, 2005
- **Banque taïwanaise** a fourni les données de 30 000 clients
- **Période d'observation** : 6 mois (avril → septembre 2005)
- **Question** : prédire si le client fera défaut le **mois suivant** (octobre)

### Pourquoi ce dataset

1. **L'intitulé "scoring"** de l'examen colle parfaitement (credit scoring = standard en banque)
2. **Ratio FN/FP bien documenté** (Bâle II/III ≈ 5:1)
3. **30 000 lignes** : assez gros pour être crédible, assez petit pour itérer vite en 1 jour
4. **Moins overused** que Telco Churn ou Home Credit
5. **Features compréhensibles** : démographie, historique paiement, factures, règlements

### Les 25 colonnes

| Catégorie | Colonnes | Description |
|-----------|----------|-------------|
| Identifiant | `ID` | À supprimer |
| Démographie | `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | Profil client |
| Historique paiement | `PAY_0` à `PAY_6` (6 mois) | Status mensuel : -2=no use, -1=paid duly, 0=revolving, 1+=retard en mois |
| Factures mensuelles | `BILL_AMT1` à `BILL_AMT6` | Montants facturés |
| Paiements mensuels | `PAY_AMT1` à `PAY_AMT6` | Montants effectivement payés |
| **Cible** | `default.payment.next.month` | 0/1 = défaut le mois suivant |

> 💡 Note : `PAY_0` correspond au plus récent (septembre). Pas de `PAY_1` — coquille du dataset.

### Constats clés

- **22% de défauts** → déséquilibre, besoin SMOTE
- **Aucune valeur manquante** ✅
- **Codes EDUCATION 0, 5, 6** non documentés (345 lignes)
- **Code MARRIAGE 0** non documenté (54 lignes)

---

## 2. Architecture globale

```
┌─────────────────────────────────────────────────────────────────────┐
│                       LOCAL DOCKER STACK                            │
│                                                                     │
│  ┌──────────────┐   ┌─────────────────┐   ┌──────────────────┐    │
│  │  PostgreSQL  │◄──┤  MLflow Server  │   │   FastAPI Server │    │
│  │  (backend)   │   │  (UI :5050)     │   │   (port :8000)   │    │
│  └──────────────┘   └─────────────────┘   └────────▲─────────┘    │
│                                                     │              │
│                                              ┌──────┴────────┐    │
│                                              │  Streamlit UI │    │
│                                              │  (port :8501) │    │
│                                              └───────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
       │
       │ git push
       ▼
┌─────────────────────────────┐
│         GITHUB              │
│  - Workflow CI (lint+tests) │
│  - Build & smoke test image │
└─────────────┬───────────────┘
              │ auto-deploy
              ▼
       ┌──────────────┐
       │   Render     │ ← API publique en prod
       │  (free tier) │
       └──────────────┘
```

---

## 3. Étape 1 — Infrastructure MLflow

### Ce qu'on a construit

- **PostgreSQL 16** : backend store de MLflow (runs, params, metrics, model registry)
- **MLflow server 2.17** : UI sur http://localhost:5050
- **Image MLflow custom** : on a ajouté `psycopg2-binary` car l'image officielle ne l'inclut pas
- **Healthcheck Python** (pas curl car absent de l'image officielle)
- **2 volumes Docker** persistants : `postgres_data`, `mlflow_artifacts`

### Justifications

| Choix | Pourquoi |
|-------|----------|
| PostgreSQL et pas SQLite | L'énoncé exige explicitement "backend Postgres ou serveur distant" |
| Docker Compose | Reproductibilité : `up -d` = stack complète opérationnelle |
| Volume nommés | Survie des données aux destructions de conteneurs |
| Réseau dédié | Communication entre services par nom DNS (`mlflow:5000`) |

### Questions probables du jury

> **Q : Pourquoi PostgreSQL et pas SQLite ?**
> R : SQLite est mono-utilisateur et ne supporte pas les transactions concurrentes. Postgres est multi-clients, supporte la production et permet la haute dispo.

> **Q : Tu as bien testé que les données persistent ?**
> R : Oui — j'ai vérifié avec `docker exec mlflow-postgres psql -U mlflow -c "SELECT * FROM experiments;"` que les expériences sont bien stockées en base après restart des conteneurs.

---

## 4. Étape 2 — Données : cleaning + feature engineering

### Le cleaning (4 opérations)

| Opération | Effet | Lignes affectées |
|-----------|-------|------------------|
| Drop `ID` | Identifiant inutile | 0 |
| Rename `default.payment.next.month` → `default` | Évite les points qui gênent en Python | 0 |
| `EDUCATION` : codes {0, 5, 6} → 4 (others) | Codes non documentés | 345 |
| `MARRIAGE` : code 0 → 3 (others) | Code non documenté | 54 |

**Total : 399 lignes modifiées (1.3%), 0 supprimée.** Cleaning conservateur.

### Le feature engineering (12 features)

Organisé en **4 thèmes business** :

#### 🅰️ Thème A — Comportement de paiement (4 features)

| Feature | Formule | Intuition |
|---------|---------|-----------|
| `PAY_DELAY_COUNT` | `Σ 1{PAY_i ≥ 1}` | Nb mois en retard sur 6 |
| `MAX_DELAY` | `max(PAY_*)` | Pire retard observé |
| `MEAN_PAY_STATUS` | `mean(PAY_*)` | Statut moyen continu |
| `HAS_EVER_DELAYED` | `1 si delay_count > 0` | Flag binaire |

#### 🅱️ Thème B — Utilisation du crédit (3 features)

| Feature | Formule | Intuition |
|---------|---------|-----------|
| `UTIL_RATIO_1` | `BILL_AMT1 / LIMIT_BAL` | Utilisation actuelle |
| `MEAN_UTIL` | `mean(BILL_i / LIMIT_BAL)` | Utilisation chronique |
| `MAX_UTIL` | `max(BILL_i / LIMIT_BAL)` | Pire mois |

#### 🅲 Thème C — Capacité de remboursement (3 features)

| Feature | Formule | Intuition |
|---------|---------|-----------|
| `TOTAL_PAID` | `Σ PAY_AMT_i` | Total payé sur 6 mois |
| `TOTAL_BILLED` | `Σ BILL_AMT_i` | Total facturé sur 6 mois |
| `PAY_TO_BILL_RATIO` | `TOTAL_PAID / TOTAL_BILLED` | Capacité de remboursement |

#### 🅳 Thème D — Tendances temporelles (2 features)

| Feature | Formule | Intuition |
|---------|---------|-----------|
| `BILL_TREND` | `(BILL_AMT1 - BILL_AMT6) / 6` | Dette monte ? |
| `PAY_TREND` | `(PAY_AMT1 - PAY_AMT6) / 6` | Paiements baissent ? |

### Validation : les features dérivées dominent

Top 5 corrélations avec la cible :
1. 🆕 `PAY_DELAY_COUNT` (0.40)
2. 🆕 `HAS_EVER_DELAYED` (0.35)
3. 🆕 `MAX_DELAY` (0.33)
4. `PAY_0` originale (0.32)
5. 🆕 `MEAN_PAY_STATUS` (0.28)

→ **4 de nos features sur 5** dans le top → le FE a clairement apporté de la valeur.

### Questions probables du jury

> **Q : Pourquoi tu as choisi de regrouper EDUCATION = 0, 5, 6 en "others" plutôt que de les supprimer ?**
> R : Supprimer 345 lignes = perdre 1.15% des données pour rien. Imputer (KNN, median) complique le pipeline sans bénéfice clair. Regrouper en "others" est défendable car ces codes étaient déjà "inclassables", ils rejoignent une catégorie existante "autres".

> **Q : Pourquoi 12 features dérivées, pas plus ?**
> R : YAGNI — on a couvert les 4 dimensions business clés (comportement, utilisation, capacité, tendance). Au-delà on tomberait dans la sur-ingénierie. Et les corrélations avec la cible montrent qu'on a pris les bonnes.

---

## 5. Étape 3 — Le score métier

### Le problème fondamental

Quand le modèle se trompe, il y a 2 types d'erreurs **aux coûts très différents** :

|  | Réalité = 0 (bon payeur) | Réalité = 1 (futur défaut) |
|---|---|---|
| **Prédit = 0** | ✅ Vrai négatif | ❌ **FN : on accorde le crédit, on perd le capital** |
| **Prédit = 1** | ❌ **FP : on refuse un bon client, manque à gagner** | ✅ Vrai positif |

### Le ratio 5:1

**Justification business** :
- Standard banking (Bâle II/III)
- 1 FN ≈ perte de capital + frais de recouvrement (~5 000-20 000 NT$)
- 1 FP ≈ manque à gagner sur intérêts (~1 000-4 000 NT$)
- Ratio ≈ 5:1 en ordre de grandeur

### Les fonctions

```python
business_cost(y_true, y_pred, fn_cost=5, fp_cost=1)
# = FP * 1 + FN * 5    (à minimiser, 0 = parfait)

business_gain(y_true, y_pred, fn_cost=5, fp_cost=1)
# = 1 - cost / worst_cost   (à maximiser, dans [0, 1])
```

### Le seuil de décision

`predict()` utilise 0.5 par défaut → optimal pour l'accuracy.
Pour notre coût asymétrique, l'optimum est **plus bas (0.28)** :
on accepte plus de FP pour éviter les FN très coûteux.

### Questions probables du jury

> **Q : Pourquoi 5:1 et pas 10:1 ?**
> R : 5:1 est le consensus de la littérature bancaire. 10:1 serait défendable pour des crédits à haut montant ou un contexte de stress. La fonction est paramétrable (`fn_cost`, `fp_cost`), donc on peut adapter sans réécrire le code.

> **Q : Comment tu as choisi le seuil 0.28 ?**
> R : Avec `find_optimal_threshold()` — on calcule le coût sur le train set pour 91 seuils (0.05 à 0.95 par 0.01) et on garde celui qui minimise le coût.

---

## 6. Étape 4 — Modèles & entraînement

### Stratégie

1. **Split 80/20 stratifié** (`stratify=y`) avant tout — pas de leakage
2. **imblearn.Pipeline** : preprocessor → SMOTE → classifier
   - `imblearn.Pipeline` (pas sklearn !) : SMOTE n'est appliqué qu'à `fit`, pas à `predict`
3. **StratifiedKFold(5)** : chaque fold contient ~22% de défauts
4. **GridSearchCV** avec `scoring=make_business_scorer(5, 1)`
5. **Optimisation du seuil** après GridSearch sur les probas du train
6. **MLflow logging** : params, metrics, modèle, confusion matrix, ROC

### Les 3 modèles comparés

| Modèle | Force | Hyperparams testés |
|--------|-------|---------------------|
| **LogReg** | Baseline interprétable | `C ∈ {0.1, 1, 10}` |
| **RandomForest** | Robuste non-linéaire | `n_estimators ∈ {200, 400}`, `max_depth ∈ {8, 16}` |
| **XGBoost** | État de l'art tabular | `n_estimators ∈ {200, 400}`, `max_depth ∈ {3, 6}`, `lr=0.1` |

### Résultats finaux

| Modèle | Test business_gain | Recall | Precision | AUC | F1 |
|--------|---------------------|--------|-----------|-----|-----|
| LogReg | 0.461 | 65% | 40% | 0.74 | 0.50 |
| RandomForest | 0.480 | 72% | 37% | **0.78** | 0.49 |
| **XGBoost** 🏆 | **0.484** | **76%** | 36% | 0.77 | 0.49 |

**Pourquoi XGBoost gagne** : recall plus élevé → moins de FN → moins coûteux.

### SHAP — top features

1. 🆕 `HAS_EVER_DELAYED` — N°1
2. `PAY_0` (originale)
3. 🆕 `TOTAL_PAID`
4. `LIMIT_BAL` (originale)
5. Démographie (SEX_2, MARRIAGE_2, EDUCATION_2)

### Questions probables du jury

> **Q : Pourquoi tu as utilisé `imblearn.Pipeline` au lieu de `sklearn.Pipeline` ?**
> R : Parce que SMOTE est un **resampler**, pas un transformer. `sklearn.Pipeline` ne sait pas que SMOTE doit être appliqué uniquement au fit, pas au predict — ça créerait du data leakage en CV. `imblearn.Pipeline` gère ça correctement.

> **Q : Pourquoi XGBoost gagne sur le business_gain alors que RandomForest a un meilleur AUC ?**
> R : AUC est indépendant du seuil de décision. Business_gain dépend du seuil. XGBoost a un meilleur **recall (75%)** que RandomForest (72%), donc moins de FN — qui coûtent 5× plus cher. C'est exactement la logique de notre métrique métier.

> **Q : Tu as fait SMOTE et class_weight en même temps ?**
> R : Non — c'est une double-correction qui empire les résultats. J'ai choisi SMOTE uniquement, et class_weight='balanced' uniquement pour la LogReg comme variante.

---

## 7. Étape 5 — API FastAPI

### Pourquoi FastAPI (pas Flask)

- ✅ Documentation auto-générée (Swagger sur `/docs`)
- ✅ Validation auto via Pydantic
- ✅ Type hints natifs
- ✅ Async possible (pas utilisé ici mais réservé)
- ✅ Plus moderne, perf > Flask

### Architecture interne

```
Client envoie 23 features RAW
       ↓
Pydantic CreditApplication (validation : types, ranges)
       ↓
clean()                ← mêmes règles que training
       ↓
engineer_features()    ← mêmes 12 features que training
       ↓
joblib pipeline (preprocessor + classifier)
       ↓
Apply optimal threshold (0.28)
       ↓
PredictionResponse {proba, threshold, prediction, label, risk_level, model_info}
```

### Points importants

- **Lifespan loader** : le modèle est chargé **une fois** au démarrage, partagé par toutes les requêtes
- **Reuse des modules `src/scoring/data.py`** : garantit que l'API applique exactement la même prep que l'entraînement
- **Pydantic** valide les inputs avant qu'ils n'atteignent le modèle (erreur 422 en cas d'input invalide)
- **CORS permissif** : permet à Streamlit (et autres origins) d'appeler l'API

### 5 endpoints

| Méthode | Route | Rôle |
|---------|-------|------|
| GET | `/` | Info service |
| GET | `/health` | Health probe (Render utilise ça) |
| GET | `/model/info` | Metadata du modèle (seuil, métriques, etc.) |
| POST | `/predict` | Prédiction unitaire |
| POST | `/predict/batch` | Prédiction en lot (max 1000) |

### Questions probables du jury

> **Q : Comment tu garantis qu'un client en production reçoit exactement le même traitement qu'à l'entraînement ?**
> R : Les fonctions `clean()` et `engineer_features()` sont dans `src/scoring/data.py` et sont **importées à la fois par le training script et par l'API**. Une seule source de vérité. Si on modifie une règle de cleaning, c'est appliqué partout en cohérence.

> **Q : Comment tu loades le modèle ?**
> R : Au démarrage de l'app via le `lifespan` context manager. Le modèle est chargé une seule fois en mémoire, pas à chaque requête → latence minimale.

---

## 8. Étape 5bis — CI/CD + déploiement Render

### CI (GitHub Actions)

À chaque push sur `main` :
1. **Job 1** : checkout, install deps Python 3.11.9, ruff lint, pytest sur tests unitaires + intégration
2. **Job 2** : build l'image Docker de l'API, lance un container, smoke test sur `/health`

→ Le déploiement n'a lieu que si **tous les tests passent** et que le **container démarre OK**.

### Déploiement Render (Blueprint)

Le fichier `render.yaml` à la racine déclare le service. Render le détecte et :
1. Build l'image via `api/Dockerfile`
2. Démarre le container avec `PORT=$PORT` (variable injectée par Render)
3. Healthcheck sur `/health` (toutes les 30s)
4. Auto-redéploie à chaque push sur `main`

### Questions probables du jury

> **Q : Pourquoi Render et pas Heroku ?**
> R : Heroku a supprimé son free tier en 2022. Render propose un free tier qui marche encore, supporte Docker nativement, et le blueprint déclaratif est plus propre que la config Heroku.

> **Q : C'est quoi la latence cold-start sur le free tier ?**
> R : Render free tier endort le service après 15 min d'inactivité. Le premier appel après ça peut prendre ~30s. Pour un démo c'est OK, en prod il faudrait un plan payant (~7$/mois).

---

## 9. Étape 6 — Streamlit

### Ce qu'on a fait

Une page unique avec **3 colonnes** :
- **Démographie** : sliders pour AGE, plafond, sélecteurs pour SEX, EDUCATION, MARRIAGE
- **Historique paiement** : 6 sliders pour PAY_0 → PAY_6
- **Factures + paiements** : 12 inputs numériques

Quand on clique **"Prédire"** :
- Appel POST vers `/predict` de l'API
- Affichage : KPIs (proba, seuil, risk_level, décision)
- **Gauge Plotly** avec le seuil métier en marqueur noir
- **Bandeau de décision** coloré (vert/rouge) avec emoji

### Pourquoi Streamlit

- Rapide à coder (~150 lignes pour cette UI)
- Re-exécution réactive du script à chaque interaction (state propre)
- Plotly natif pour les gauges
- Déployable indépendamment de l'API

### Questions probables du jury

> **Q : Pourquoi tu n'as pas mis Streamlit et l'API dans le même service ?**
> R : Séparation des responsabilités. L'API est l'API, elle peut être appelée par n'importe quel client (Streamlit, mobile, autre service). Streamlit est une UI parmi d'autres possibles. Architecture découplée = chacun évolue indépendamment.

---

## 10. Étape 7 — Data drift avec Evidently

### Pourquoi monitorer

Un modèle entraîné sur 2005 sera moins bon en 2026 si la clientèle a changé :
- Vieillissement → distribution `AGE` décale
- Inflation → `LIMIT_BAL` augmente
- Crise économique → plus de retards

**Sans monitoring on découvre le problème après dégradation.**
**Avec monitoring on a un signal avant.**

### Comment on détecte

| Type | Test |
|------|------|
| Numérique | **Wasserstein distance** (normalisée), seuil 0.1 |
| Catégorielle | **Chi²**, p-value < 0.05 |

Dataset drift "détecté" si > 50% des colonnes dérivent.

### Notre simulation

On a "fabriqué" du drift réaliste sur 3 features :
- AGE +3 ans
- LIMIT_BAL +20%
- PAY_0 plus de retards

→ Evidently a **précisément détecté ces 3 features** comme dérivées. ✅

### Stratégie de réentraînement

```
Cron quotidien :
  1. Snapshot prédictions des 30 derniers jours
  2. Comparer reference (train original) vs current (30j)
  3. Si drift détecté :
       - Notifier
       - Lancer training automatique (GitHub Action)
       - Si nouveau modèle améliore business_gain > +2% → promouvoir
       - Sinon → rollback automatique
```

### Questions probables du jury

> **Q : Pourquoi Wasserstein et pas KS ?**
> R : Wasserstein (Earth Mover's Distance) mesure le "coût" de transformer une distribution en l'autre. Plus stable que KS sur petits échantillons. C'est le défaut Evidently.

> **Q : Tu as monitoré le drift sur des vraies données prod ?**
> R : Non — comme on ne peut pas avoir de "vraies" données 2026, j'ai simulé un drift réaliste (vieillissement + inflation + plus de retards) pour démontrer le pipeline de détection. En prod réelle on aurait un cron qui compare le train original avec un snapshot rolling de 30 jours.

---

## 11. Commandes essentielles

```bash
# Démarrer toute la stack locale
docker compose up -d --build

# Entraîner / réentraîner
python -m scoring.train

# Lancer les tests
pytest tests/ -v

# Vérifier le drift
python -m scoring.drift

# Reconstruire un notebook
python notebooks/_build_eda.py            # ou _build_business_score, _build_training, _build_drift
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --inplace

# Arrêter toute la stack
docker compose down

# Voir les logs d'un service
docker compose logs -f api
```

---

## 12. FAQ — autres questions probables du jury

### Sur l'architecture

> **Q : Pourquoi tu as choisi cette structure de dossiers ?**
> R : Convention Python moderne — `src/` pour le module métier réutilisable, `api/` pour le service web, `streamlit_app/` pour l'UI, `tests/` pour tous les tests pytest, `notebooks/` pour l'exploration, `infra/` pour les Dockerfiles d'infra (MLflow), `docs/` pour la documentation et les figures. Chaque dossier a une responsabilité claire.

### Sur la reproductibilité

> **Q : Comment tu garantis la reproductibilité ?**
> R : 4 niveaux :
> 1. Versions Python pinnées dans `requirements.txt` (==, pas >=)
> 2. Random seeds fixés partout (`random_state=42`)
> 3. Docker (Python 3.11.9 baseline garantie)
> 4. CI sur GitHub Actions reproductible

### Sur le code

> **Q : Tu testes quoi exactement ?**
> R : 31 tests organisés en classes :
> - **business_score** (15 tests) : coût, gain, scorer sklearn, threshold optimization, custom ratios
> - **preprocessing** (5 tests) : column splitting, transformer build, unknown categories
> - **api** (11 tests) : endpoints meta, validation Pydantic, prédiction low/high risk, batch

### Sur la prod

> **Q : Si on veut passer en prod réelle, qu'est-ce qu'on rajoute ?**
> R :
> 1. Authentification API (Bearer token ou OAuth)
> 2. Rate limiting
> 3. Logging structuré + tracing (OpenTelemetry)
> 4. Monitoring Prometheus + Grafana
> 5. MLflow Model Registry avec stages (dev → staging → prod)
> 6. Database pour stocker les prédictions (drift réel + audit)
> 7. A/B testing entre champion et challenger

---

## 13. Glossaire des termes clés

| Terme | Définition |
|-------|------------|
| **MLOps** | DevOps appliqué au ML — pipelines reproductibles, monitoring, déploiement automatisé |
| **Data leakage** | Quand de l'info du test set "fuit" dans le train. Bug subtile, métriques gonflées artificiellement |
| **SMOTE** | Synthetic Minority Over-sampling Technique. Crée des points synthétiques de la classe minoritaire en interpolant entre voisins |
| **GridSearchCV** | Cherche les meilleurs hyperparams en testant toutes les combinaisons via cross-validation |
| **StratifiedKFold** | Variante de KFold qui préserve la proportion des classes dans chaque fold |
| **Business score** | Métrique custom qui pondère les erreurs selon leur coût métier réel |
| **Threshold optimization** | Trouver le seuil de probabilité qui maximise la métrique métier (pas l'accuracy) |
| **SHAP** | Attribution de la contribution de chaque feature à une prédiction (basé sur la théorie des jeux) |
| **MLflow tracking** | Logger params/metrics/modèles/artefacts pour chaque run d'expérimentation |
| **Data drift** | Évolution de la distribution des features en prod vs au training |
| **Wasserstein distance** | Mesure de distance entre distributions (= "coût" de transformation) |
| **Cross-validation** | Évaluer un modèle en répétant train/validation sur plusieurs splits |
| **Confusion matrix** | Tableau TP/FP/TN/FN, base de toutes les métriques de classification |
| **Recall / Sensibilité** | TP / (TP + FN) — proportion de positifs réels qu'on capture |
| **Precision** | TP / (TP + FP) — proportion de prédictions positives qui sont correctes |
| **ROC AUC** | Aire sous la courbe ROC, indépendant du seuil, mesure la capacité discriminante |
| **Pipeline** | Enchaînement de transformations + modèle, fit/predict atomique |
| **OHE** | One-Hot Encoding — transformer une variable catégorielle en colonnes binaires |
| **CI/CD** | Continuous Integration / Deployment — automatisation tests + déploiement |
| **Blueprint Render** | Fichier YAML déclaratif pour configurer le déploiement Render automatiquement |

---

## 14. Pour terminer

Tu as un projet **end-to-end** qui couvre :
- ✅ **Toutes** les étapes de l'examen (1 → 7)
- ✅ **Choix techniques justifiables** business + technique
- ✅ **Reproductibilité** garantie (Docker, seeds, requirements pinnés)
- ✅ **Tests automatisés** (31 tests, tous PASSED)
- ✅ **Documentation complète** (notebooks pédagogiques + slides + README + ce guide)

### Avant la soutenance

1. ✅ Relire ce guide (focus sur les **questions probables**)
2. ✅ Ouvrir les 4 notebooks et survoler les figures clés
3. ✅ Déployer sur Render (10 min) pour avoir une URL live
4. ✅ Faire un screenshot Streamlit (pour les slides)
5. ✅ Préparer 2-3 phrases pour ton **pitch initial** (point 0)

### Pendant la soutenance

- 🎤 Commence par le **pitch en 30s** (point 0)
- 🖼️ Montre **le schéma d'architecture** (point 2)
- 🎯 Insiste sur le **score métier custom** — c'est ton meilleur argument différenciant
- 📊 Si tu peux montrer **Streamlit en live**, c'est très impactant visuellement

Bonne soutenance ! 🎓
