from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scoring_app.config import REFERENCE_PATH
from scoring_app.model_io import load_reference_profile


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(reference.quantile(quantiles).to_numpy())
    if len(edges) < 3:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = np.maximum(ref_counts / max(ref_counts.sum(), 1), 1e-6)
    cur_pct = np.maximum(cur_counts / max(cur_counts.sum(), 1), 1e-6)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def analyze_drift(
    current_data: pd.DataFrame,
    reference_path: Path = REFERENCE_PATH,
) -> pd.DataFrame:
    profile = load_reference_profile(reference_path)
    rows = []
    for column in profile["columns"]:
        if column not in current_data.columns:
            rows.append({"feature": column, "psi": None, "status": "missing"})
            continue
        ref_summary = profile["summary"][column]
        synthetic_reference = pd.Series(
            np.linspace(ref_summary["min"], ref_summary["max"], len(current_data))
        )
        psi = population_stability_index(synthetic_reference, current_data[column])
        status = "alert" if psi >= 0.25 else "watch" if psi >= 0.10 else "ok"
        rows.append({"feature": column, "psi": psi, "status": status})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze data drift with PSI.")
    parser.add_argument("current_csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/drift_report.json"))
    args = parser.parse_args()

    report = analyze_drift(pd.read_csv(args.current_csv))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.to_json(orient="records", indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "alerts": int((report.status == "alert").sum())}))


if __name__ == "__main__":
    main()
