"""Build the comprehensive EDA notebook programmatically.

The notebook is the central document of the EDA phase: every cleaning rule
and every engineered feature is derived **inline** with business
justification, formula, code, and validation. The src/scoring/data.py
module replicates these rules for production reuse (training + API).
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


# =============================================================================
# Title
# =============================================================================
md(
    """
# 🔬 EDA, Cleaning & Feature Engineering — UCI Credit Card Default

Ce notebook est le **document de référence** pour la préparation des données.
Tout y est inline et documenté : chaque règle de nettoyage et chaque feature
dérivée a sa justification métier, sa formule, son code et sa validation
statistique.

**Dataset** : 30 000 clients de cartes de crédit taïwanais (Avril → Septembre
2005), 25 colonnes. Cible binaire : `default.payment.next.month`.

**Plan du notebook**

1. Setup
2. Vue structurelle des données brutes
3. Qualité des données — détection des anomalies
4. ✂️ **Cleaning détaillé** — 4 opérations avec before/after
5. Statistiques descriptives (post-cleaning)
6. Analyse de la cible
7. Analyse univariée
8. Analyse bivariée (variable vs cible) avec tests stats
9. Analyse multivariée (corrélations)
10. Détection des outliers
11. Analyses temporelles (6 mois)
12. 🛠️ **Feature engineering détaillé** — 12 features avec justification
13. Validation de l'utilité des nouvelles features
14. Sauvegarde du dataset processed
"""
)

# =============================================================================
# Section 1 — Setup
# =============================================================================
md("## 1. ⚙️ Setup")

code(
    """
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style='whitegrid', context='notebook')
pd.set_option('display.max_columns', 60)
pd.set_option('display.width', 200)

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
FIG_DIR = ROOT / 'docs' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

RAW_PATH = ROOT / 'data' / 'raw' / 'UCI_Credit_Card.csv'
PROCESSED_PATH = ROOT / 'data' / 'processed' / 'credit_clean.parquet'

print(f'ROOT      : {ROOT}')
print(f'RAW_PATH  : {RAW_PATH.relative_to(ROOT)}')
print(f'numpy     : {np.__version__}')
print(f'pandas    : {pd.__version__}')
"""
)

# =============================================================================
# Section 2 — Vue structurelle
# =============================================================================
md(
    """
## 2. Vue structurelle des données brutes

Objectif : comprendre la forme du dataset, les types, la cardinalité.
"""
)

code(
    """
df_raw = pd.read_csv(RAW_PATH)
print(f'Shape : {df_raw.shape}')
print(f'Memory: {df_raw.memory_usage(deep=True).sum() / 1024**2:.2f} MB')
df_raw.head()
"""
)

code("df_raw.dtypes.value_counts()")
code("df_raw.info()")

code(
    """
# Cardinalité (nombre de valeurs uniques par colonne)
df_raw.nunique().sort_values()
"""
)

code(
    """
print('Doublons exactes :', df_raw.duplicated().sum())
"""
)

# =============================================================================
# Section 3 — Qualité (anomalies)
# =============================================================================
md(
    """
## 3. Qualité des données — détection des anomalies

Trois choses à vérifier :
- Valeurs manquantes
- **Codes catégoriels en dehors de la documentation officielle**
- Valeurs incohérentes (BILL_AMT négatif, etc.)
"""
)

code(
    """
# Valeurs manquantes
missing = df_raw.isnull().sum()
print(f'Total NaN : {missing.sum()}')
if missing.sum() > 0:
    print(missing[missing > 0])
"""
)

md(
    """
### 3.1 — Codes EDUCATION et MARRIAGE non documentés

D'après la documentation UCI :
- `EDUCATION` doit être dans `{1, 2, 3, 4}` (graduate school / university /
  high school / others)
- `MARRIAGE` doit être dans `{1, 2, 3}` (married / single / others)

Vérifions ce qu'on a vraiment dans les données brutes.
"""
)

code(
    """
print('--- EDUCATION (valeurs brutes) ---')
print(df_raw['EDUCATION'].value_counts().sort_index())
print()
print('--- MARRIAGE (valeurs brutes) ---')
print(df_raw['MARRIAGE'].value_counts().sort_index())
"""
)

md(
    """
**Anomalies détectées :**
- `EDUCATION` contient les codes `0`, `5`, `6` qui ne sont pas dans la doc
  (345 clients soit 1.15% des données)
- `MARRIAGE` contient le code `0` (54 clients, 0.18%)

Ces codes sont vraisemblablement des erreurs de saisie ou des cas
inclassables. Nous les regrouperons dans la catégorie « others » lors
du cleaning (Section 4).
"""
)

md("### 3.2 — Codes PAY_* (historique de paiement)")

code(
    """
PAY_COLS = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
for col in PAY_COLS:
    print(f'{col} : {sorted(df_raw[col].unique())}')
"""
)

md(
    """
**Lecture des codes PAY_* :**
- `-2` : pas d'utilisation ce mois-ci (no use)
- `-1` : payé en totalité, à l'heure (paid duly)
- `0` : crédit revolving utilisé mais pas en retard
- `1` à `8` : retard de 1 à 8+ mois

Les codes `-2` et `0` ne sont pas dans la doc officielle mais sont
cohérents avec le métier (consensus Kaggle). On les **garde tels quels** :
ce sont des valeurs ordinales légitimes pour les modèles.
"""
)

md("### 3.3 — Valeurs négatives sur BILL_AMT")

code(
    """
BILL_COLS = [f'BILL_AMT{i}' for i in range(1, 7)]
neg_count = (df_raw[BILL_COLS] < 0).sum()
neg_pct_any = (df_raw[BILL_COLS] < 0).any(axis=1).mean() * 100
print('Lignes avec BILL_AMT négatif par mois :')
print(neg_count)
print(f'\\n% de clients avec au moins un BILL négatif : {neg_pct_any:.1f}%')
"""
)

md(
    """
**Interprétation** : un `BILL_AMT` négatif n'est pas une anomalie — cela
correspond à un avoir (le client a payé plus que la facture le mois
précédent, ou la banque a effectué un remboursement). Légitime, on garde.
"""
)

# =============================================================================
# Section 4 — Cleaning détaillé
# =============================================================================
md(
    """
## 4. ✂️ Cleaning détaillé

On applique **4 opérations** sur le DataFrame brut. Chaque opération est
expliquée, justifiée et validée par un before/after.
"""
)

md(
    """
### 4.1 — Opération 1 : drop de la colonne `ID`

**Pourquoi ?** `ID` est un identifiant client (1, 2, 3, ...). Aucune info
prédictive — pire, il pourrait introduire un biais spurieux. À supprimer
systématiquement.
"""
)

code(
    """
# Snapshot AVANT
print(f'AVANT : df_raw a {df_raw.shape[1]} colonnes')
print(f'        ID range : {df_raw["ID"].min()} → {df_raw["ID"].max()}')
"""
)

code(
    """
df = df_raw.drop(columns=['ID']).copy()
print(f'APRÈS : df a {df.shape[1]} colonnes')
"""
)

md(
    """
### 4.2 — Opération 2 : rename de la cible

**Pourquoi ?** Le nom `default.payment.next.month` contient des **points**
qui empêchent l'accès attribut-style (`df.default` ne marche pas avec
des points dans le nom). On renomme en `default`.
"""
)

code(
    """
# Snapshot AVANT
print('AVANT :', [c for c in df.columns if 'default' in c])
"""
)

code(
    """
df = df.rename(columns={'default.payment.next.month': 'default'})
TARGET = 'default'
print('APRÈS :', [c for c in df.columns if 'default' in c])
"""
)

md(
    """
### 4.3 — Opération 3 : recodage `EDUCATION`

**Règle** : les codes `0`, `5`, `6` (non documentés) sont remappés vers
`4` (« others »).

**Pourquoi ce choix plutôt qu'imputer ou supprimer ?**
- Supprimer 345 lignes (1.15%) = perdre de l'info pour rien
- Imputer (KNN, median) complique le pipeline et n'apporte rien ici
- Regrouper en `others` est défendable : ces codes étaient déjà
  « inclassables », ils rejoignent une catégorie « autres ».
"""
)

code(
    """
print('AVANT recodage :')
print(df['EDUCATION'].value_counts().sort_index())
"""
)

code(
    """
df.loc[~df['EDUCATION'].isin([1, 2, 3, 4]), 'EDUCATION'] = 4

print('APRÈS recodage :')
print(df['EDUCATION'].value_counts().sort_index())

moved_edu = ((df_raw['EDUCATION'] == 0).sum()
             + (df_raw['EDUCATION'] == 5).sum()
             + (df_raw['EDUCATION'] == 6).sum())
print(f'\\n→ {moved_edu} clients déplacés vers EDUCATION=4 (others)')
"""
)

md(
    """
### 4.4 — Opération 4 : recodage `MARRIAGE`

**Règle** : le code `0` (non documenté, 54 clients) est remappé vers
`3` (« others »).
"""
)

code(
    """
print('AVANT recodage :')
print(df['MARRIAGE'].value_counts().sort_index())
"""
)

code(
    """
df.loc[~df['MARRIAGE'].isin([1, 2, 3]), 'MARRIAGE'] = 3

print('APRÈS recodage :')
print(df['MARRIAGE'].value_counts().sort_index())

moved_mar = (df_raw['MARRIAGE'] == 0).sum()
print(f'\\n→ {moved_mar} clients déplacés vers MARRIAGE=3 (others)')
"""
)

md(
    """
### 4.5 — Synthèse du cleaning

| Opération | Effet | Lignes affectées |
|-----------|-------|------------------|
| Drop `ID` | 25 → 24 colonnes | 0 |
| Rename target | `default.payment.next.month` → `default` | 0 |
| Recodage `EDUCATION` | 7 codes → 4 codes | 345 |
| Recodage `MARRIAGE` | 4 codes → 3 codes | 54 |

Au total **399 lignes modifiées (1.3%)** et **0 ligne supprimée**.
Le cleaning est volontairement conservateur.
"""
)

code(
    """
print(f'Dataset clean : {df.shape}')
df.head()
"""
)

# =============================================================================
# Section 5 — Statistiques descriptives
# =============================================================================
md(
    """
## 5. Statistiques descriptives (post-cleaning)
"""
)

code("df.describe().T")

code(
    """
# Quantiles étendus pour repérer les queues extrêmes
df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
"""
)

code(
    """
# Skewness — |skew| > 1 indique une distribution très asymétrique
skew = df.skew().sort_values(ascending=False)
print('Top 10 variables les plus asymétriques :')
print(skew.head(10).round(2))
"""
)

# =============================================================================
# Section 6 — Analyse de la cible
# =============================================================================
md(
    """
## 6. Analyse de la cible

Constat clé : **22% de défauts** → dataset déséquilibré → besoin de
SMOTE ou `class_weight='balanced'` à la phase d'entraînement.
"""
)

code(
    """
target_counts = df[TARGET].value_counts()
target_pct = df[TARGET].value_counts(normalize=True) * 100
print(target_counts)
print()
print(target_pct.round(2).astype(str) + ' %')
"""
)

code(
    """
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(x=TARGET, data=df, ax=axes[0])
axes[0].set_title('Effectifs par classe')
for p in axes[0].patches:
    axes[0].annotate(int(p.get_height()),
                     (p.get_x() + p.get_width() / 2, p.get_height()),
                     ha='center', va='bottom')
axes[1].pie(target_counts, labels=['Non-défaut (0)', 'Défaut (1)'],
            autopct='%1.1f%%', startangle=90, colors=['#4C72B0', '#DD8452'])
axes[1].set_title('Répartition')
plt.tight_layout()
plt.savefig(FIG_DIR / '04_target_balance.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

# =============================================================================
# Section 7 — Univariate
# =============================================================================
md(
    """
## 7. Analyse univariée

### 7.A — Variables numériques continues
"""
)

code(
    """
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
sns.histplot(df['LIMIT_BAL'], kde=True, ax=axes[0, 0], bins=40, color='#4C72B0')
axes[0, 0].set_title('Distribution LIMIT_BAL')
sns.boxplot(x=df['LIMIT_BAL'], ax=axes[0, 1], color='#4C72B0')
axes[0, 1].set_title('Boxplot LIMIT_BAL')
sns.histplot(df['AGE'], kde=True, ax=axes[1, 0], bins=30, color='#55A868')
axes[1, 0].set_title('Distribution AGE')
sns.boxplot(x=df['AGE'], ax=axes[1, 1], color='#55A868')
axes[1, 1].set_title('Boxplot AGE')
plt.tight_layout()
plt.savefig(FIG_DIR / '05a_limit_age.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

code(
    """
PAY_AMT_COLS = [f'PAY_AMT{i}' for i in range(1, 7)]

fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, col in zip(axes.ravel(), BILL_COLS):
    sns.histplot(np.log1p(df[col].clip(lower=0)), kde=True, ax=ax, bins=40)
    ax.set_title(f'log1p({col})')
plt.suptitle('Distributions des BILL_AMT (log1p)', y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / '05a_bill_amt_log.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

md("### 7.B — Variables catégorielles")

code(
    """
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ['SEX', 'EDUCATION', 'MARRIAGE']):
    sns.countplot(x=col, data=df, ax=ax)
    ax.set_title(f'Effectifs par {col}')
plt.tight_layout()
plt.savefig(FIG_DIR / '05b_categoricals.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

code(
    """
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, col in zip(axes.ravel(), PAY_COLS):
    sns.countplot(x=col, data=df, ax=ax)
    ax.set_title(f'{col}')
plt.suptitle('Distribution des PAY_* — historique de paiement', y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / '05b_pay_status.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

# =============================================================================
# Section 8 — Bivariate
# =============================================================================
md(
    """
## 8. Analyse bivariée (variable vs cible)

On veut identifier les variables dont la distribution diffère entre
clients défaut et non-défaut.
"""
)

md("### 8.A — Boxplots numérique vs cible")

code(
    """
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, col in zip(axes.ravel(), ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'PAY_AMT1']):
    sns.boxplot(x=TARGET, y=col, data=df, ax=ax)
    ax.set_title(f'{col} selon {TARGET}')
    ax.set_xlabel('default (0 = non, 1 = oui)')
plt.tight_layout()
plt.savefig(FIG_DIR / '06a_boxplots_target.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

md(
    """
### 8.B — Test statistique : Mann-Whitney U

Test non-paramétrique : les deux groupes (défaut vs non-défaut) ont-ils
la même distribution sur une variable numérique ?
"""
)

code(
    """
from scipy.stats import mannwhitneyu
print(f'{"Variable":<15} {"U-stat":>15} {"p-value":>15} {"significatif":>15}')
print('-' * 75)
for col in ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'PAY_AMT1', 'PAY_0']:
    g0 = df.loc[df[TARGET] == 0, col]
    g1 = df.loc[df[TARGET] == 1, col]
    u, p = mannwhitneyu(g0, g1, alternative='two-sided')
    sig = 'OUI' if p < 0.001 else 'non'
    print(f'{col:<15} {u:>15.0f} {p:>15.2e} {sig:>15}')
"""
)

md("### 8.C — Taux de défaut par catégorie")

code(
    """
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ['SEX', 'EDUCATION', 'MARRIAGE']):
    rate = df.groupby(col)[TARGET].mean().sort_values()
    sns.barplot(x=rate.index, y=rate.values, ax=ax)
    ax.set_ylabel('taux de défaut')
    ax.set_title(f'Taux de défaut par {col}')
    ax.axhline(df[TARGET].mean(), color='red', linestyle='--',
               label=f'moyenne globale ({df[TARGET].mean():.1%})')
    ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / '06b_default_rate_by_cat.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

code(
    """
from scipy.stats import chi2_contingency
print('Test du Chi² (H0: indépendance entre la variable et la cible)')
for col in ['SEX', 'EDUCATION', 'MARRIAGE']:
    table = pd.crosstab(df[col], df[TARGET])
    chi2, p, dof, _ = chi2_contingency(table)
    sig = 'OUI' if p < 0.001 else 'non'
    print(f'{col:<15} chi2={chi2:>8.2f}  p={p:.2e}  significatif={sig}')
"""
)

md("### 8.D — Taux de défaut selon PAY_0 (variable star)")

code(
    """
rate_by_pay0 = df.groupby('PAY_0')[TARGET].agg(['mean', 'count']).round(3)
rate_by_pay0.columns = ['taux_defaut', 'effectif']
print(rate_by_pay0)
"""
)

code(
    """
plt.figure(figsize=(10, 5))
rate = df.groupby('PAY_0')[TARGET].mean()
sns.barplot(x=rate.index.astype(str), y=rate.values, palette='YlOrRd')
plt.axhline(df[TARGET].mean(), color='gray', ls='--',
            label=f'moyenne globale ({df[TARGET].mean():.1%})')
plt.title('Taux de défaut selon PAY_0 (mois le plus récent)')
plt.ylabel('taux de défaut')
plt.xlabel('PAY_0 (-2=no use, -1=paid duly, 0=revolving, 1+=delay)')
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / '06b_default_by_pay0.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

# =============================================================================
# Section 9 — Multivariate
# =============================================================================
md("## 9. Analyse multivariée")

code(
    """
corr = df.corr(method='pearson')
plt.figure(figsize=(14, 11))
sns.heatmap(corr, cmap='coolwarm', center=0, annot=False, square=True,
            cbar_kws={'shrink': 0.8})
plt.title('Heatmap des corrélations (Pearson)')
plt.tight_layout()
plt.savefig(FIG_DIR / '07_corr_heatmap.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

code(
    """
print('Top 15 variables corrélées avec default (Pearson) :')
print(corr[TARGET].drop(TARGET).abs().sort_values(ascending=False).head(15).round(3))
"""
)

code(
    """
plt.figure(figsize=(7, 5))
sns.heatmap(df[PAY_COLS].corr(), annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True)
plt.title('Corrélations PAY_0..PAY_6 (persistance des retards)')
plt.tight_layout()
plt.savefig(FIG_DIR / '07_corr_pay_block.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

# =============================================================================
# Section 10 — Outliers
# =============================================================================
md(
    """
## 10. Détection des outliers

On signale les outliers mais on les **garde** : XGBoost et RandomForest y sont
robustes. La LogReg aura un `RobustScaler` dans son pipeline.
"""
)

code(
    """
num_cols = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'PAY_AMT1']
outliers_pct = {}
for col in num_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (df[col] < low) | (df[col] > high)
    outliers_pct[col] = mask.mean() * 100
pd.Series(outliers_pct).round(2).to_frame('% outliers IQR')
"""
)

code(
    """
fig, axes = plt.subplots(3, 4, figsize=(16, 9))
cols_box = ['LIMIT_BAL', 'AGE'] + BILL_COLS[:5] + PAY_AMT_COLS[:5]
for ax, col in zip(axes.ravel(), cols_box):
    sns.boxplot(x=df[col], ax=ax, color='#4C72B0')
    ax.set_title(col, fontsize=10)
plt.tight_layout()
plt.savefig(FIG_DIR / '08_boxplots_grid.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

# =============================================================================
# Section 11 — Temporal
# =============================================================================
md(
    """
## 11. Analyses temporelles (6 mois)

Mapping : `PAY_0` = septembre 2005 (le plus récent), `PAY_6` = avril 2005.
Idem pour `BILL_AMT1` (sept.) → `BILL_AMT6` (avril).
"""
)

code(
    """
months = ['Sept (PAY_0)', 'Août (PAY_2)', 'Juillet (PAY_3)',
          'Juin (PAY_4)', 'Mai (PAY_5)', 'Avril (PAY_6)']

bill_mean = df[BILL_COLS].mean()
pay_mean = df[PAY_AMT_COLS].mean()
bill_mean.index = months
pay_mean.index = months

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
bill_mean.plot(kind='line', marker='o', ax=axes[0], color='#4C72B0')
axes[0].set_title('Moyenne BILL_AMT par mois (NT$)')
axes[0].grid(alpha=0.3)
pay_mean.plot(kind='line', marker='o', ax=axes[1], color='#C44E52')
axes[1].set_title('Moyenne PAY_AMT par mois (NT$)')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / '09_temporal_means.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

code(
    """
delay_pct = (df[PAY_COLS] >= 1).mean() * 100
delay_pct.index = months
plt.figure(figsize=(10, 4))
sns.barplot(x=delay_pct.index, y=delay_pct.values, palette='YlOrRd')
plt.title('% de clients en retard par mois')
plt.ylabel('% en retard (PAY >= 1)')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(FIG_DIR / '09_delay_pct_per_month.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

# =============================================================================
# Section 12 — FEATURE ENGINEERING DÉTAILLÉ
# =============================================================================
md(
    """
## 12. 🛠️ Feature engineering détaillé

On construit **12 features dérivées** organisées en **4 thèmes métier** :

| Thème | Features | Intuition business |
|-------|----------|--------------------|
| **A — Comportement de paiement** | `PAY_DELAY_COUNT`, `MAX_DELAY`, `MEAN_PAY_STATUS`, `HAS_EVER_DELAYED` | Résume l'historique de retards sur 6 mois |
| **B — Utilisation du crédit** | `UTIL_RATIO_1`, `MEAN_UTIL`, `MAX_UTIL` | À quel point le client utilise son plafond |
| **C — Capacité de remboursement** | `TOTAL_PAID`, `TOTAL_BILLED`, `PAY_TO_BILL_RATIO` | Le client paye-t-il proportionnellement à ce qu'il doit |
| **D — Tendances temporelles** | `BILL_TREND`, `PAY_TREND` | La dette monte ? Les paiements baissent ? |

Pour chaque feature on présente : **intuition métier → formule → code → validation**.
"""
)

# ---------- Thème A ----------
md(
    """
### 12.A — Thème : comportement de paiement

#### A.1 `PAY_DELAY_COUNT` — Nombre de mois en retard sur 6

- **Intuition** : un client en retard 1 fois sur 6 n'est pas comme un
  client en retard 5 fois sur 6. On compte les mois avec retard.
- **Formule** : `PAY_DELAY_COUNT = Σ 1{PAY_X ≥ 1}` pour les 6 mois
- **Range** : 0 (jamais en retard) à 6 (toujours en retard)
"""
)

code(
    """
df['PAY_DELAY_COUNT'] = (df[PAY_COLS] >= 1).sum(axis=1).astype(int)

print('Distribution PAY_DELAY_COUNT :')
print(df['PAY_DELAY_COUNT'].value_counts().sort_index())
print()
print('Taux de défaut par PAY_DELAY_COUNT :')
print(df.groupby('PAY_DELAY_COUNT')[TARGET].mean().round(3))
"""
)

md(
    """
#### A.2 `MAX_DELAY` — Pire retard observé sur 6 mois

- **Intuition** : un défaut sévère même une seule fois est très révélateur.
- **Formule** : `MAX_DELAY = max(PAY_0, PAY_2, …, PAY_6)`
- **Range** : -2 (pas d'utilisation) à 8 (retard 8+ mois)
"""
)

code(
    """
df['MAX_DELAY'] = df[PAY_COLS].max(axis=1)
print(df['MAX_DELAY'].value_counts().sort_index())
print()
print('Taux de défaut par MAX_DELAY :')
print(df.groupby('MAX_DELAY')[TARGET].mean().round(3))
"""
)

md(
    """
#### A.3 `MEAN_PAY_STATUS` — Statut de paiement moyen

- **Intuition** : un score continu qui capture le comportement global.
  Plus c'est positif, plus le client est chroniquement en retard.
- **Formule** : `MEAN_PAY_STATUS = mean(PAY_0, PAY_2, …, PAY_6)`
"""
)

code(
    """
df['MEAN_PAY_STATUS'] = df[PAY_COLS].mean(axis=1)
print(df['MEAN_PAY_STATUS'].describe().round(3))
"""
)

md(
    """
#### A.4 `HAS_EVER_DELAYED` — Flag binaire

- **Intuition** : version simplifiée — a-t-il jamais été en retard ?
- **Formule** : `HAS_EVER_DELAYED = 1 si PAY_DELAY_COUNT > 0 sinon 0`
- **Utilité** : interprétable et simple, peut suffire pour LogReg.
"""
)

code(
    """
df['HAS_EVER_DELAYED'] = (df['PAY_DELAY_COUNT'] > 0).astype(int)
print('Taux de défaut selon HAS_EVER_DELAYED :')
print(df.groupby('HAS_EVER_DELAYED')[TARGET].mean().round(3))
"""
)

code(
    """
# Validation visuelle Thème A
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, col in zip(axes.ravel(),
                   ['PAY_DELAY_COUNT', 'MAX_DELAY',
                    'MEAN_PAY_STATUS', 'HAS_EVER_DELAYED']):
    sns.boxplot(x=TARGET, y=col, data=df, ax=ax)
    ax.set_title(f'{col} selon default')
plt.suptitle('Thème A — Comportement de paiement', y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / '12a_theme_payment.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

# ---------- Thème B ----------
md(
    """
### 12.B — Thème : utilisation du crédit

Un client qui utilise 95% de son plafond est plus risqué qu'un client à 20%.
On calcule des **ratios d'utilisation** (utilisation rate = bill / limit).
"""
)

md(
    """
#### B.1 `UTIL_RATIO_1` — Utilisation au mois le plus récent

- **Intuition** : utilisation actuelle, le signal le plus frais.
- **Formule** : `UTIL_RATIO_1 = BILL_AMT1 / LIMIT_BAL`
"""
)

code(
    """
# Protection contre division par 0 (LIMIT_BAL ne devrait jamais être 0, mais on protège)
limit_safe = df['LIMIT_BAL'].replace(0, np.nan)
df['UTIL_RATIO_1'] = (df['BILL_AMT1'] / limit_safe).fillna(0)
print(df['UTIL_RATIO_1'].describe().round(3))
"""
)

md(
    """
#### B.2 `MEAN_UTIL` — Utilisation moyenne sur 6 mois

- **Intuition** : utilisation chronique (vs ponctuelle).
- **Formule** : `MEAN_UTIL = mean(BILL_AMT_i / LIMIT_BAL)` pour i=1..6
"""
)

code(
    """
util_block = df[BILL_COLS].div(limit_safe, axis=0)
df['MEAN_UTIL'] = util_block.mean(axis=1).fillna(0)
print(df['MEAN_UTIL'].describe().round(3))
"""
)

md(
    """
#### B.3 `MAX_UTIL` — Pire mois d'utilisation

- **Intuition** : a-t-il déjà flirté avec son plafond ?
- **Formule** : `MAX_UTIL = max(BILL_AMT_i / LIMIT_BAL)` pour i=1..6
"""
)

code(
    """
df['MAX_UTIL'] = util_block.max(axis=1).fillna(0)
print(df['MAX_UTIL'].describe().round(3))
"""
)

code(
    """
# Validation Thème B
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ['UTIL_RATIO_1', 'MEAN_UTIL', 'MAX_UTIL']):
    sns.boxplot(x=TARGET, y=col, data=df, ax=ax, showfliers=False)
    ax.set_title(f'{col} selon default')
plt.suptitle('Thème B — Utilisation du crédit (outliers cachés)', y=1.05, fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / '12b_theme_utilization.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

# ---------- Thème C ----------
md(
    """
### 12.C — Thème : capacité de remboursement

On combine factures et paiements pour mesurer la capacité du client
à honorer ses engagements globalement sur 6 mois.
"""
)

md(
    """
#### C.1 `TOTAL_PAID` — Total payé sur 6 mois

- **Formule** : `TOTAL_PAID = Σ PAY_AMT_i` pour i=1..6
"""
)

code(
    """
df['TOTAL_PAID'] = df[PAY_AMT_COLS].sum(axis=1)
print(df['TOTAL_PAID'].describe().round(0))
"""
)

md(
    """
#### C.2 `TOTAL_BILLED` — Total facturé sur 6 mois

- **Formule** : `TOTAL_BILLED = Σ BILL_AMT_i` pour i=1..6
"""
)

code(
    """
df['TOTAL_BILLED'] = df[BILL_COLS].sum(axis=1)
print(df['TOTAL_BILLED'].describe().round(0))
"""
)

md(
    """
#### C.3 `PAY_TO_BILL_RATIO` — Capacité de remboursement

- **Intuition** : 1.0 = il paie exactement ce qu'il doit. < 1.0 = il
  accumule de la dette. > 1.0 = il rembourse plus que sa facture
  (rare, souvent dû à un avoir).
- **Formule** : `TOTAL_PAID / TOTAL_BILLED`
- **Protection** : on évite la division par 0 ou par négatif.
"""
)

code(
    """
billed_safe = df['TOTAL_BILLED'].where(df['TOTAL_BILLED'] > 0, np.nan)
df['PAY_TO_BILL_RATIO'] = (df['TOTAL_PAID'] / billed_safe).fillna(0)
print(df['PAY_TO_BILL_RATIO'].describe().round(3))
print()
print('Médiane par classe :')
print(df.groupby(TARGET)['PAY_TO_BILL_RATIO'].median().round(3))
"""
)

code(
    """
# Validation Thème C (log scale car très skewé)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ['TOTAL_PAID', 'TOTAL_BILLED', 'PAY_TO_BILL_RATIO']):
    sns.boxplot(x=TARGET, y=col, data=df, ax=ax, showfliers=False)
    ax.set_title(f'{col} selon default')
plt.suptitle('Thème C — Capacité de remboursement', y=1.05, fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / '12c_theme_repayment.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

# ---------- Thème D ----------
md(
    """
### 12.D — Thème : tendances temporelles

Capturer si la dette monte (BILL_AMT augmente) ou si les paiements
baissent (PAY_AMT diminue) sur les 6 mois.

**Convention** : `BILL_AMT1` = septembre (le plus récent), `BILL_AMT6`
= avril (le plus ancien). Donc `BILL_AMT1 - BILL_AMT6` > 0 signifie
**la dette a augmenté** sur la période.
"""
)

md(
    """
#### D.1 `BILL_TREND` — Pente de la dette sur 6 mois

- **Intuition** : valeur positive = dette qui monte = mauvais signe.
- **Formule** : `BILL_TREND = (BILL_AMT1 - BILL_AMT6) / 6`
"""
)

code(
    """
df['BILL_TREND'] = (df['BILL_AMT1'] - df['BILL_AMT6']) / 6
print(df['BILL_TREND'].describe().round(0))
print()
print('Médiane par classe :')
print(df.groupby(TARGET)['BILL_TREND'].median().round(0))
"""
)

md(
    """
#### D.2 `PAY_TREND` — Pente des paiements sur 6 mois

- **Intuition** : valeur positive = paye de plus en plus = bon signe.
- **Formule** : `PAY_TREND = (PAY_AMT1 - PAY_AMT6) / 6`
"""
)

code(
    """
df['PAY_TREND'] = (df['PAY_AMT1'] - df['PAY_AMT6']) / 6
print(df['PAY_TREND'].describe().round(0))
print()
print('Médiane par classe :')
print(df.groupby(TARGET)['PAY_TREND'].median().round(0))
"""
)

code(
    """
# Validation Thème D
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, col in zip(axes, ['BILL_TREND', 'PAY_TREND']):
    sns.boxplot(x=TARGET, y=col, data=df, ax=ax, showfliers=False)
    ax.set_title(f'{col} selon default')
plt.suptitle('Thème D — Tendances temporelles', y=1.05, fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / '12d_theme_trends.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

# =============================================================================
# Section 13 — Validation finale
# =============================================================================
md(
    """
## 13. Validation finale — les nouvelles features sont-elles utiles ?

On compare la corrélation des features dérivées avec la cible vs les
features originales. Si les nouvelles features dominent le classement,
on a bien fait notre travail.
"""
)

code(
    """
print(f'Dataset enrichi : {df.shape}')
new_cols = ['PAY_DELAY_COUNT', 'MAX_DELAY', 'MEAN_PAY_STATUS', 'HAS_EVER_DELAYED',
            'UTIL_RATIO_1', 'MEAN_UTIL', 'MAX_UTIL',
            'TOTAL_PAID', 'TOTAL_BILLED', 'PAY_TO_BILL_RATIO',
            'BILL_TREND', 'PAY_TREND']
print(f'{len(new_cols)} features ajoutées')
"""
)

code(
    """
# Top 20 corrélations (abs) avec default — features originales + dérivées mélangées
corr_full = df.corr()[TARGET].drop(TARGET).abs().sort_values(ascending=False)
top20 = corr_full.head(20).round(3)

# Marquer les features dérivées
labels = ['🆕 ' + c if c in new_cols else '   ' + c for c in top20.index]
print('Top 20 variables corrélées avec default :')
for label, val in zip(labels, top20.values):
    print(f'  {val:.3f}  {label}')
"""
)

code(
    """
# Visualisation du top 20
plt.figure(figsize=(10, 8))
colors = ['#DD8452' if c in new_cols else '#4C72B0' for c in top20.index]
sns.barplot(x=top20.values, y=top20.index, palette=colors)
plt.title('Top 20 — corrélation absolue avec default\\n(orange = features dérivées)')
plt.xlabel('|corrélation Pearson|')
plt.tight_layout()
plt.savefig(FIG_DIR / '13_feature_importance_corr.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

code(
    """
# Validation Spearman (robuste aux outliers)
corr_s = df.corr(method='spearman')[TARGET].drop(TARGET).abs().sort_values(ascending=False)
top15_s = corr_s.head(15).round(3)
print('Top 15 corrélations Spearman (abs) avec default :')
for c, v in top15_s.items():
    tag = '🆕' if c in new_cols else '  '
    print(f'  {v:.3f}  {tag} {c}')
"""
)

# =============================================================================
# Section 14 — Sauvegarde
# =============================================================================
md(
    """
## 14. Sauvegarde du dataset processed
"""
)

code(
    """
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(PROCESSED_PATH, index=False)
print(f'Saved → {PROCESSED_PATH.relative_to(ROOT)}')
print(f'Shape : {df.shape}')
print(f'Size  : {PROCESSED_PATH.stat().st_size / 1024:.0f} KB')
"""
)

# =============================================================================
# Section 15 — Synthèse + lien vers la prod
# =============================================================================
md(
    """
## 15. 📌 Synthèse

### Cleaning appliqué
| Opération | Effet |
|-----------|-------|
| Drop `ID` | 25 → 24 colonnes |
| Rename `default.payment.next.month` → `default` | nom Python-friendly |
| EDUCATION : recoder {0, 5, 6} → 4 (others) | 345 lignes |
| MARRIAGE : recoder 0 → 3 (others) | 54 lignes |

### Feature engineering — 12 nouvelles colonnes

| Thème | Features | Description |
|-------|----------|-------------|
| A. Paiement | PAY_DELAY_COUNT, MAX_DELAY, MEAN_PAY_STATUS, HAS_EVER_DELAYED | Historique de retards |
| B. Utilisation | UTIL_RATIO_1, MEAN_UTIL, MAX_UTIL | Bill / Limit |
| C. Remboursement | TOTAL_PAID, TOTAL_BILLED, PAY_TO_BILL_RATIO | Capacité globale |
| D. Tendances | BILL_TREND, PAY_TREND | Évolution 6 mois |

### Dataset final
- **Shape** : 30 000 lignes × 36 colonnes (24 cleaned + 12 engineered)
- **Sauvegarde** : `data/processed/credit_clean.parquet`

### Réutilisation en production

Les mêmes règles sont encapsulées dans `src/scoring/data.py` :

```python
from scoring.data import prepare
df = prepare()   # load_raw → clean → engineer_features
```

Ce module est appelé par le script d'entraînement (`src/scoring/train.py`)
et par l'API FastAPI au moment de la prédiction → **garantie qu'un
client reçoit en prod exactement le même traitement qu'à l'entraînement**.

### Prochaine étape

→ Étape 3 de l'examen : définir la **fonction de score métier**
(coût FN vs coût FP).
"""
)


# =============================================================================
# Build
# =============================================================================
def build() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3.11 (.venv)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.9",
        },
    }
    out = Path(__file__).resolve().parents[1] / "notebooks" / "01_eda.ipynb"
    with open(out, "w") as f:
        nbf.write(nb, f)
    print(f"Built {out}")
    print(f"Cells: {len(cells)}")


if __name__ == "__main__":
    build()
