"""Build the EDA notebook programmatically with nbformat.

Run once, then execute with `jupyter nbconvert --to notebook --execute`.
This script is part of the project so anyone can regenerate the notebook.
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
# EDA — UCI Credit Card Default

**Dataset** : 30 000 clients de cartes de crédit taïwanais observés sur 6 mois
(avril → septembre 2005).
**Cible** : `default` (1 = défaut de paiement le mois suivant, 0 sinon).

Ce notebook couvre 10 sections :
1. Vue structurelle
2. Qualité des données
3. Statistiques descriptives
4. Analyse de la cible
5. Univariate analysis
6. Bivariate analysis (variable vs cible)
7. Multivariate (corrélations, heatmaps)
8. Outliers
9. Analyses temporelles (6 mois)
10. Feature engineering & dataset processed

Les cleaning rules et le feature engineering sont définis dans
`src/scoring/data.py` afin de garantir la cohérence entre EDA, training et API.
"""
)

# =============================================================================
# Setup
# =============================================================================
md("## ⚙️ Setup")

code(
    """
import sys
from pathlib import Path

# Make src importable
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT / 'src'))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style='whitegrid', context='notebook')
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

FIG_DIR = ROOT / 'docs' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

from scoring.data import (
    load_raw, clean, decode_categoricals, engineer_features, prepare,
    PAY_COLS, BILL_COLS, PAY_AMT_COLS, TARGET,
)

df_raw = load_raw()
df = clean(df_raw)
print(f'Raw   : {df_raw.shape}')
print(f'Clean : {df.shape}')
df.head()
"""
)

# =============================================================================
# Section 1 — Vue structurelle
# =============================================================================
md(
    """
## 1️⃣ Vue structurelle

**Objectif** : comprendre la forme et le contenu brut du dataset avant
toute analyse.
"""
)

code("print('Shape :', df.shape)\nprint('Memory:', df.memory_usage(deep=True).sum() / 1024**2, 'MB')")
code("df.dtypes.value_counts()")
code("df.info()")
code("df.nunique().sort_values()")
code("print('Doublons :', df.duplicated().sum())")
code(
    """
# Vue rapide : 3 lignes aléatoires
df.sample(3, random_state=0)
"""
)

# =============================================================================
# Section 2 — Qualité des données
# =============================================================================
md(
    """
## 2️⃣ Qualité des données

**Objectif** : repérer valeurs manquantes, codes anormaux, valeurs
incohérentes. Le dataset UCI a quelques codes non documentés sur
`EDUCATION` (0, 5, 6) et `MARRIAGE` (0) — la fonction `clean()` les
remappe vers la modalité "others".
"""
)

code(
    """
missing = df.isnull().sum()
print('Total missing values :', missing.sum())
missing[missing > 0]
"""
)

code(
    """
# Codes catégoriels après cleaning
print('EDUCATION :', sorted(df['EDUCATION'].unique()))
print('MARRIAGE  :', sorted(df['MARRIAGE'].unique()))
print('SEX       :', sorted(df['SEX'].unique()))
"""
)

code(
    """
# Avant/après cleaning sur EDUCATION
print('Avant cleaning :')
print(df_raw['EDUCATION'].value_counts().sort_index())
print()
print('Après cleaning :')
print(df['EDUCATION'].value_counts().sort_index())
"""
)

code(
    """
# Codes PAY (status historique)
for col in PAY_COLS:
    print(f'{col} : {sorted(df[col].unique())}')
"""
)

code(
    """
# Valeurs négatives sur BILL_AMT (peut signaler un avoir/remboursement)
neg_bill = (df[BILL_COLS] < 0).sum()
print('Lignes avec BILL_AMT négatif (par mois) :')
print(neg_bill)
print()
print(f'% de clients avec au moins un BILL négatif : '
      f'{((df[BILL_COLS] < 0).any(axis=1).mean() * 100):.1f}%')
"""
)

# =============================================================================
# Section 3 — Statistiques descriptives
# =============================================================================
md(
    """
## 3️⃣ Statistiques descriptives

**Objectif** : valeurs typiques, dispersion, asymétrie, queues lourdes.
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
# Skewness (asymétrie) — |skew| > 1 indique une distribution très asymétrique
skew = df.drop(columns=[TARGET]).skew().sort_values(ascending=False)
print('Top 10 variables les plus asymétriques :')
print(skew.head(10))
"""
)

code(
    """
# Statistiques séparées par classe de la cible
df.groupby(TARGET).describe().T
"""
)

# =============================================================================
# Section 4 — La cible
# =============================================================================
md(
    """
## 4️⃣ Analyse de la cible

**Constat** : ~22% de défauts → dataset **déséquilibré** → nécessite SMOTE
ou `class_weight='balanced'` lors du training.
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

# Bar plot
sns.countplot(x=TARGET, data=df, ax=axes[0])
axes[0].set_title('Effectifs par classe')
for p in axes[0].patches:
    axes[0].annotate(int(p.get_height()),
                     (p.get_x() + p.get_width() / 2, p.get_height()),
                     ha='center', va='bottom')

# Pie chart
axes[1].pie(target_counts, labels=['Non-défaut (0)', 'Défaut (1)'],
            autopct='%1.1f%%', startangle=90, colors=['#4C72B0', '#DD8452'])
axes[1].set_title('Répartition')

plt.tight_layout()
plt.savefig(FIG_DIR / '04_target_balance.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

# =============================================================================
# Section 5 — Univariate analysis
# =============================================================================
md(
    """
## 5️⃣ Analyse univariée

### 5.A — Variables numériques continues

Histogrammes + boxplots pour `LIMIT_BAL`, `AGE`, `BILL_AMT*`, `PAY_AMT*`.
"""
)

code(
    """
# LIMIT_BAL et AGE
fig, axes = plt.subplots(2, 2, figsize=(13, 8))

sns.histplot(df['LIMIT_BAL'], kde=True, ax=axes[0, 0], bins=40, color='#4C72B0')
axes[0, 0].set_title('Distribution LIMIT_BAL (plafond de crédit)')
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
# BILL_AMT (en log car très skewé) — grille 2x3
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

code(
    """
# PAY_AMT (idem)
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, col in zip(axes.ravel(), PAY_AMT_COLS):
    sns.histplot(np.log1p(df[col].clip(lower=0)), kde=True, ax=ax, bins=40, color='#C44E52')
    ax.set_title(f'log1p({col})')
plt.suptitle('Distributions des PAY_AMT (log1p)', y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / '05a_pay_amt_log.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

md("### 5.B — Variables catégorielles")

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
# Distribution des PAY_* (status historique paiement)
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, col in zip(axes.ravel(), PAY_COLS):
    sns.countplot(x=col, data=df, ax=ax)
    ax.set_title(f'{col} (status mois)')
plt.suptitle('Distribution des PAY_* — historique de paiement', y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / '05b_pay_status.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

# =============================================================================
# Section 6 — Bivariate
# =============================================================================
md(
    """
## 6️⃣ Analyse bivariée (variable vs cible)

**Objectif** : repérer quelles variables ont une **distribution
différente selon que le client a fait défaut ou non** — ce sont les
candidates fortes pour la prédiction.
"""
)

md("### 6.A — Numérique vs cible (boxplots)")

code(
    """
num_cols_compare = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'PAY_AMT1']
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, col in zip(axes.ravel(), num_cols_compare):
    sns.boxplot(x=TARGET, y=col, data=df, ax=ax)
    ax.set_title(f'{col} selon {TARGET}')
    ax.set_xlabel('default (0 = non, 1 = oui)')
plt.tight_layout()
plt.savefig(FIG_DIR / '06a_boxplots_target.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

code(
    """
# Densités superposées (KDE) — voir le chevauchement des distributions
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.kdeplot(data=df, x='LIMIT_BAL', hue=TARGET, common_norm=False, ax=axes[0])
axes[0].set_title('Densité LIMIT_BAL par classe')
sns.kdeplot(data=df, x='AGE', hue=TARGET, common_norm=False, ax=axes[1])
axes[1].set_title('Densité AGE par classe')
plt.tight_layout()
plt.savefig(FIG_DIR / '06a_kde_target.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

code(
    """
# Test statistique Mann-Whitney (non-paramétrique) — variables numériques clés
from scipy.stats import mannwhitneyu
print('Test de Mann-Whitney U (H0: distributions identiques entre classes)')
print('-' * 75)
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

md("### 6.B — Catégorielle vs cible (taux de défaut)")

code(
    """
# Taux de défaut par catégorie démographique
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
# Table croisée SEX × default (en %)
print('SEX vs default (% par ligne) :')
print(pd.crosstab(df['SEX'], df[TARGET], normalize='index').round(3) * 100)
print()
print('EDUCATION vs default (% par ligne) :')
print(pd.crosstab(df['EDUCATION'], df[TARGET], normalize='index').round(3) * 100)
print()
print('MARRIAGE vs default (% par ligne) :')
print(pd.crosstab(df['MARRIAGE'], df[TARGET], normalize='index').round(3) * 100)
"""
)

code(
    """
# Test du Chi² — dépendance significative ?
from scipy.stats import chi2_contingency
print('Test du Chi² (H0: indépendance entre la variable et la cible)')
print('-' * 65)
for col in ['SEX', 'EDUCATION', 'MARRIAGE']:
    table = pd.crosstab(df[col], df[TARGET])
    chi2, p, dof, _ = chi2_contingency(table)
    sig = 'OUI' if p < 0.001 else 'non'
    print(f'{col:<15} chi2={chi2:>8.2f}  p={p:.2e}  significatif={sig}')
"""
)

code(
    """
# Taux de défaut par PAY_0 (le mois le plus récent)
rate_by_pay0 = df.groupby('PAY_0')[TARGET].agg(['mean', 'count']).round(3)
rate_by_pay0.columns = ['taux_defaut', 'effectif']
print('Taux de défaut par PAY_0 (status mois le plus récent) :')
print(rate_by_pay0)
print()
print('→ La progression du taux de défaut avec le retard valide '
      "qu'on tient une variable très prédictive.")
"""
)

code(
    """
plt.figure(figsize=(10, 5))
rate = df.groupby('PAY_0')[TARGET].mean()
sns.barplot(x=rate.index.astype(str), y=rate.values, palette='YlOrRd')
plt.axhline(df[TARGET].mean(), color='gray', ls='--',
            label=f'moyenne globale ({df[TARGET].mean():.1%})')
plt.title('Taux de défaut selon PAY_0 (status du mois le plus récent)')
plt.ylabel('taux de défaut')
plt.xlabel('PAY_0 (-2=no use, -1=paid duly, 0=revolving, 1+=delay months)')
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / '06b_default_by_pay0.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

# =============================================================================
# Section 7 — Multivariate
# =============================================================================
md(
    """
## 7️⃣ Analyse multivariée

**Objectif** : repérer corrélations entre variables (multicolinéarité)
et les plus liées à la cible.
"""
)

code(
    """
# Matrice de corrélation Pearson
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
# Top corrélations avec la cible
corr_target = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False)
print('Top 15 variables les plus corrélées (en valeur absolue) avec default :')
print(corr_target.head(15).round(3))
"""
)

code(
    """
# Corrélation Spearman (basée sur les rangs, plus robuste aux outliers)
corr_s = df.corr(method='spearman')
corr_s_target = corr_s[TARGET].drop(TARGET).sort_values()
plt.figure(figsize=(10, 8))
sns.barplot(x=corr_s_target.values, y=corr_s_target.index,
            palette='coolwarm')
plt.axvline(0, color='black', lw=0.5)
plt.title('Corrélation Spearman de chaque variable avec default')
plt.xlabel('coefficient de Spearman')
plt.tight_layout()
plt.savefig(FIG_DIR / '07_corr_spearman_target.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

code(
    """
# Corrélations entre PAY_* (devraient être fortement positives :
# les retards sont persistants dans le temps)
plt.figure(figsize=(7, 5))
sns.heatmap(df[PAY_COLS].corr(), annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True)
plt.title('Corrélations entre PAY_0..PAY_6 (persistance des retards)')
plt.tight_layout()
plt.savefig(FIG_DIR / '07_corr_pay_block.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

# =============================================================================
# Section 8 — Outliers
# =============================================================================
md(
    """
## 8️⃣ Détection des outliers

**Objectif** : identifier les valeurs extrêmes susceptibles de perturber
les modèles linéaires. Nous gardons les outliers (XGBoost et RandomForest
y sont robustes), mais on les signale pour le standardisation côté
LogisticRegression.
"""
)

code(
    """
# Méthode IQR — % d'outliers par variable numérique clé
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
# Z-score : % de points |z| > 3
zscore_pct = {}
for col in num_cols:
    z = np.abs((df[col] - df[col].mean()) / df[col].std())
    zscore_pct[col] = (z > 3).mean() * 100
pd.Series(zscore_pct).round(2).to_frame('% outliers Z>3')
"""
)

code(
    """
# Grille de boxplots pour visualisation rapide
fig, axes = plt.subplots(3, 4, figsize=(16, 9))
cols_box = (['LIMIT_BAL', 'AGE'] + BILL_COLS[:5] + PAY_AMT_COLS[:5])
for ax, col in zip(axes.ravel(), cols_box):
    sns.boxplot(x=df[col], ax=ax, color='#4C72B0')
    ax.set_title(col, fontsize=10)
plt.tight_layout()
plt.savefig(FIG_DIR / '08_boxplots_grid.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

# =============================================================================
# Section 9 — Analyses temporelles
# =============================================================================
md(
    """
## 9️⃣ Analyses temporelles (6 mois)

**Objectif** : exploiter la dimension temporelle (avril → septembre 2005).
Indices : tendance des factures, persistance des retards, % de clients
en retard chaque mois.
"""
)

code(
    """
# Mapping mois → label lisible (PAY_0 = septembre, PAY_6 = avril)
months = ['Sept (PAY_0)', 'Août (PAY_2)', 'Juillet (PAY_3)',
          'Juin (PAY_4)', 'Mai (PAY_5)', 'Avril (PAY_6)']

# Évolution de la moyenne BILL_AMT et PAY_AMT
bill_mean = df[BILL_COLS].mean()
pay_mean = df[PAY_AMT_COLS].mean()
bill_mean.index = months
pay_mean.index = months

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
bill_mean.plot(kind='line', marker='o', ax=axes[0], color='#4C72B0')
axes[0].set_title('Moyenne BILL_AMT par mois (NT$)')
axes[0].set_ylabel('moyenne BILL_AMT')
axes[0].grid(alpha=0.3)

pay_mean.plot(kind='line', marker='o', ax=axes[1], color='#C44E52')
axes[1].set_title('Moyenne PAY_AMT par mois (NT$)')
axes[1].set_ylabel('moyenne PAY_AMT')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / '09_temporal_means.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

code(
    """
# % de clients en retard (PAY_X >= 1) chaque mois
delay_pct = (df[PAY_COLS] >= 1).mean() * 100
delay_pct.index = months
plt.figure(figsize=(10, 4))
sns.barplot(x=delay_pct.index, y=delay_pct.values, palette='YlOrRd')
plt.title('% de clients en retard de paiement par mois')
plt.ylabel('% en retard (PAY >= 1)')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(FIG_DIR / '09_delay_pct_per_month.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

# =============================================================================
# Section 10 — Feature engineering & dataset processed
# =============================================================================
md(
    """
## 🛠️ Feature engineering

Les 12 features dérivées créées dans `engineer_features()` (voir
`src/scoring/data.py`).
"""
)

code(
    """
df_full = engineer_features(df)
new_cols = [c for c in df_full.columns if c not in df.columns]
print(f'{len(new_cols)} features ajoutées :')
for c in new_cols:
    print(f'  - {c}')

df_full[new_cols].describe().T
"""
)

code(
    """
# Distribution des nouvelles features les plus parlantes par classe
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, col in zip(axes.ravel(),
                   ['PAY_DELAY_COUNT', 'MAX_DELAY',
                    'MEAN_PAY_STATUS', 'MEAN_UTIL']):
    sns.boxplot(x=TARGET, y=col, data=df_full, ax=ax)
    ax.set_title(f'{col} selon default')
plt.tight_layout()
plt.savefig(FIG_DIR / '10_engineered_features.png', dpi=120, bbox_inches='tight')
plt.show()
"""
)

code(
    """
# Top corrélations des features engineered avec la cible
corr_full = df_full.corr()[TARGET].drop(TARGET).abs().sort_values(ascending=False)
print('Top 15 variables (cleaned + engineered) corrélées avec default :')
print(corr_full.head(15).round(3))
"""
)

code(
    """
# Sauvegarde du dataset processed
out = ROOT / 'data' / 'processed' / 'credit_clean.parquet'
out.parent.mkdir(parents=True, exist_ok=True)
df_full.to_parquet(out, index=False)
print(f'Saved → {out.relative_to(ROOT)}')
print(f'Shape : {df_full.shape}')
"""
)

# =============================================================================
# Summary
# =============================================================================
md(
    """
## 📌 Synthèse des findings

| # | Finding |
|---|---------|
| 1 | Dataset **propre** : aucune valeur manquante, mais quelques codes non documentés sur EDUCATION et MARRIAGE → traités |
| 2 | Cible **déséquilibrée** : 22% de défauts → SMOTE ou class_weight indispensable |
| 3 | Les variables **`PAY_*`** (historique de paiement) sont **de très loin les plus prédictives** |
| 4 | `LIMIT_BAL` et `AGE` montrent des différences statistiquement significatives entre classes (Mann-Whitney p < 1e-50) |
| 5 | Les tests **Chi²** confirment que `SEX`, `EDUCATION`, `MARRIAGE` sont liées à la cible (p < 0.001) |
| 6 | Forte **corrélation entre BILL_AMT successifs** (~0.95) → multicolinéarité → préférer des features agrégées |
| 7 | Le dataset est **stable dans le temps** : pas de tendance majeure sur 6 mois (utile pour la baseline drift) |
| 8 | **12 features dérivées** créées : MEAN_PAY_STATUS, PAY_DELAY_COUNT, MEAN_UTIL, etc. — capturent le comportement global |
| 9 | Dataset processed sauvegardé dans `data/processed/credit_clean.parquet` (36 colonnes, 30 000 lignes) |

Prochaine étape : définir la **fonction de score métier** (Étape 3 de l'examen).
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
    out = Path(__file__).parent / "01_eda.ipynb"
    with open(out, "w") as f:
        nbf.write(nb, f)
    print(f"Built {out}")


if __name__ == "__main__":
    build()
