# scripts/

Builders qui génèrent les notebooks Jupyter du projet.

| Script | Notebook produit |
|--------|------------------|
| `build_eda_notebook.py` | `notebooks/01_eda.ipynb` |
| `build_business_score_notebook.py` | `notebooks/02_business_score.ipynb` |
| `build_training_notebook.py` | `notebooks/03_training.ipynb` |
| `build_drift_notebook.py` | `notebooks/04_drift.ipynb` |
| `build_all_notebooks.py` | les 4 d'un coup (avec exécution) |

## Usage

```bash
# Reconstruire un notebook précis (sans l'exécuter)
.venv/bin/python scripts/build_eda_notebook.py

# Tout reconstruire + exécuter
.venv/bin/python scripts/build_all_notebooks.py

# Tout reconstruire sans exécuter (rapide)
.venv/bin/python scripts/build_all_notebooks.py --no-run
```

Pourquoi des builders ? Les notebooks `.ipynb` sont du JSON peu lisible
en diff git. On édite la version Python, on regénère le notebook quand
on veut le re-livrer.
