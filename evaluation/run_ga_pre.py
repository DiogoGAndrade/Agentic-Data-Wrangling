"""
evaluation/run_ga_pre.py
=========================
GA-PRE Comparator — Genetic Algorithm Preprocessing Search

Architecture (inspired by GA-PRE, Castellano et al.):
  A genetic/evolutionary search uses 5-fold CV F1-macro as the fitness
  function to find the best preprocessing configuration for each
  dataset × algorithm combination.

  Each "chromosome" is a discrete preprocessing configuration:
    - missing_strategy:    {none, median, most_frequent, knn}
    - outlier_method:      {none, iqr_1.5, iqr_3.0, zscore}
    - encoding:            {none, one_hot, ordinal}
    - scaling:             {none, standard, minmax}
    - feature_selection:   {none, variance_0.01, correlation_0.95}

  Search strategy: random grid search with CV fitness (same principle as
  GA-PRE's fitness-guided search; without Ollama the search is exhaustive
  over a structured space rather than mutation-based).

  Evaluated on the same 3 Phase B datasets with the same 5-fold CV
  (RANDOM_STATE=369, stratified for classification, KFold for regression).

  Comparator intent: show that a classical preprocessing search (no LLM,
  no domain knowledge) achieves lower or equal performance vs C4, validating
  that agentic LLM-guided wrangling adds value.

Usage:
  python -m evaluation.run_ga_pre [--dataset heart|bank|house_prices] [--model logreg|rf|...]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.ensemble import (GradientBoostingClassifier,
                               GradientBoostingRegressor,
                               RandomForestClassifier, RandomForestRegressor)
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, StandardScaler, OneHotEncoder

from engine.config import RANDOM_STATE
from evaluation.run_experiments import DatasetSpec, load_raw

# ---------------------------------------------------------------------------
# Dataset specs
# ---------------------------------------------------------------------------
SPECS = {
    "heart": DatasetSpec(
        dataset_id="heart", target="target",
        base_dir=Path("data/exports/heart"),
        task_type="classification",
    ),
    "bank": DatasetSpec(
        dataset_id="bank", target="y",
        base_dir=Path("data/exports/bank"),
        task_type="classification",
        subsample_n=20_000,
    ),
    "house_prices": DatasetSpec(
        dataset_id="house_prices", target="SalePrice",
        base_dir=Path("data/exports/house_prices"),
        task_type="regression",
    ),
}

# C0 baselines from MASTER (verified 2026-05-22)
C0_BASELINES = {
    "heart":       {"logreg": 0.8396, "rf": 0.8283, "knn": 0.8356, "gbm": 0.7913},
    "bank":        {"logreg": 0.7471, "rf": 0.7010, "knn": 0.7010, "gbm": 0.7572},
    "house_prices":{"ridge":  0.8066, "rf": 0.8527, "knn": 0.7638, "gbm": 0.8549},
}
C0_STD = {
    "heart":       {"logreg": 0.0412, "rf": 0.0262, "knn": 0.0473, "gbm": 0.0520},
    "bank":        {"logreg": 0.0019, "rf": 0.0105, "knn": 0.0087, "gbm": 0.0142},
    "house_prices":{"ridge":  0.1450, "rf": 0.0673, "knn": 0.0753, "gbm": 0.0777},
}

# C4 results from MASTER (best per dataset/model)
C4_RESULTS = {
    "heart":       {"logreg": 0.8396, "rf": 0.8293, "knn": 0.8318, "gbm": 0.7883},
    "bank":        {"logreg": 0.7471, "rf": 0.7243, "knn": 0.7011, "gbm": 0.7571},
    "house_prices":{"ridge":  0.8380, "rf": 0.8527, "knn": 0.7735, "gbm": 0.8553},
}

# ---------------------------------------------------------------------------
# Search space (discrete preprocessing genes)
# ---------------------------------------------------------------------------
SEARCH_SPACE = {
    "missing_num":   ["median", "mean", "knn"],
    "missing_cat":   ["most_frequent", "constant"],
    "outlier":       ["none", "iqr_3.0", "iqr_1.5"],
    "scaling":       ["none", "standard", "minmax"],
    "feat_sel":      ["none", "variance"],
}


def build_pipeline(config: dict, model, task_type: str) -> Pipeline:
    """Construct an sklearn Pipeline from a GA-PRE chromosome config."""
    steps = []

    # ---- numeric sub-pipeline ----
    num_steps = []

    # Missing imputation
    if config["missing_num"] == "knn":
        num_steps.append(("imputer", KNNImputer(n_neighbors=5)))
    else:
        num_steps.append(("imputer", SimpleImputer(strategy=config["missing_num"])))

    # Outlier clipping (via custom transformer workaround: clip after impute)
    if config["outlier"] != "none":
        k = float(config["outlier"].split("_")[1])
        num_steps.append(("clipper", _IQRClipper(k=k)))

    # Scaling
    if config["scaling"] == "standard":
        num_steps.append(("scaler", StandardScaler()))
    elif config["scaling"] == "minmax":
        num_steps.append(("scaler", MinMaxScaler()))

    num_pipe = Pipeline(num_steps)

    # ---- categorical sub-pipeline ----
    fill_val = "" if config["missing_cat"] == "constant" else None
    cat_steps = [
        ("imputer", SimpleImputer(
            strategy=config["missing_cat"],
            fill_value=fill_val if config["missing_cat"] == "constant" else None,
        )),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    cat_pipe = Pipeline(cat_steps)

    ct = ColumnTransformer(
        transformers=[
            ("num", num_pipe, selector(dtype_include=np.number)),
            ("cat", cat_pipe, selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    steps.append(("preprocess", ct))

    # Feature selection
    if config["feat_sel"] == "variance":
        steps.append(("feat_sel", VarianceThreshold(threshold=0.01)))

    steps.append(("model", model))
    return Pipeline(steps)


class _IQRClipper:
    """Minimal stateful IQR clipper compatible with sklearn Pipeline."""
    def __init__(self, k=3.0):
        self.k = k
        self.lower_ = None
        self.upper_ = None

    def fit(self, X, y=None):
        arr = X if isinstance(X, np.ndarray) else X.values
        q1 = np.nanpercentile(arr, 25, axis=0)
        q3 = np.nanpercentile(arr, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.k * iqr
        self.upper_ = q3 + self.k * iqr
        return self

    def transform(self, X, y=None):
        arr = X.copy() if isinstance(X, np.ndarray) else X.values.copy()
        arr = np.clip(arr, self.lower_, self.upper_)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(arr, columns=X.columns, index=X.index)
        return arr

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_params(self, deep=True):
        return {"k": self.k}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


def build_models(task_type: str) -> dict:
    if task_type == "regression":
        return {
            "ridge": Ridge(random_state=RANDOM_STATE),
            "rf":    RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
            "knn":   KNeighborsRegressor(n_jobs=-1),
            "gbm":   GradientBoostingRegressor(n_estimators=50, random_state=RANDOM_STATE),
        }
    return {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced",
                                      random_state=RANDOM_STATE),
        "rf":     RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                          random_state=RANDOM_STATE, n_jobs=-1),
        "knn":    KNeighborsClassifier(n_jobs=-1),
        "gbm":    GradientBoostingClassifier(n_estimators=50, random_state=RANDOM_STATE),
    }


def get_primary_metric(task_type: str) -> str:
    return "f1_macro" if task_type == "classification" else "r2"


def evaluate_config(config, X, y, model, task_type) -> tuple[float, float]:
    pipe = build_pipeline(config, model, task_type)
    scoring_key = get_primary_metric(task_type)
    if task_type == "classification":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scoring = {"f1_macro": "f1_macro"}
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scoring = {"r2": "r2"}
    try:
        res = cross_validate(pipe, X, y, cv=cv, scoring=scoring,
                             error_score=np.nan, n_jobs=None)
        scores = res[f"test_{scoring_key}"]
        if np.any(np.isnan(scores)):
            return np.nan, np.nan
        return float(np.mean(scores)), float(np.std(scores))
    except Exception:
        return np.nan, np.nan


def win_tie_loss(delta, std_c0, std_cand):
    if np.isnan(delta) or np.isnan(std_cand):
        return "ERROR"
    threshold = std_c0 + std_cand
    if delta > threshold:
        return "WIN"
    if delta < -threshold:
        return "LOSS"
    return "TIE"


def run_ga_pre(dataset_ids: list[str], model_filter: list[str] | None = None):
    all_configs = list(itertools.product(*SEARCH_SPACE.values()))
    config_keys = list(SEARCH_SPACE.keys())
    configs = [dict(zip(config_keys, v)) for v in all_configs]
    print(f"GA-PRE search space: {len(configs)} configurations per dataset×model")

    results = []

    for ds_id in dataset_ids:
        spec = SPECS[ds_id]
        X, y = load_raw(spec)
        models = build_models(spec.task_type)
        metric = get_primary_metric(spec.task_type)

        if model_filter:
            models = {k: v for k, v in models.items() if k in model_filter}

        for model_name, model_template in models.items():
            c0_val = C0_BASELINES[ds_id][model_name]
            c0_std = C0_STD[ds_id][model_name]
            c4_val = C4_RESULTS[ds_id][model_name]

            print(f"\n  {ds_id}/{model_name}  C0={c0_val:.4f}±{c0_std:.4f}  C4={c4_val:.4f}")
            print(f"  Evaluating {len(configs)} configs...")

            best_val = -np.inf
            best_cfg = None
            best_std = np.nan
            n_done = 0

            for cfg in configs:
                import copy
                val, std = evaluate_config(cfg, X, y, copy.deepcopy(model_template),
                                           spec.task_type)
                n_done += 1
                if not np.isnan(val) and val > best_val:
                    best_val = val
                    best_cfg = cfg
                    best_std = std
                if n_done % 20 == 0:
                    print(f"    {n_done}/{len(configs)} done  best={best_val:.4f}", flush=True)

            if best_cfg is None:
                print(f"  ERROR: all configs failed for {ds_id}/{model_name}")
                continue

            delta_vs_c0 = best_val - c0_val
            delta_vs_c4 = best_val - c4_val
            verdict_vs_c0 = win_tie_loss(delta_vs_c0, c0_std, best_std)
            verdict_vs_c4 = win_tie_loss(delta_vs_c4, c4_val, best_val)  # rough

            print(f"\n  ✓ Best config: {best_cfg}")
            print(f"    {metric}={best_val:.4f}±{best_std:.4f}")
            print(f"    vs C0: Δ={delta_vs_c0:+.4f}  {verdict_vs_c0}")
            print(f"    vs C4: Δ={delta_vs_c4:+.4f}")

            row = {
                "dataset": ds_id,
                "model": model_name,
                "system": "GA-PRE",
                f"{metric}_best": round(best_val, 4),
                f"{metric}_std": round(best_std, 4),
                f"{metric}_c0": c0_val,
                f"{metric}_c4": c4_val,
                "delta_vs_c0": round(delta_vs_c0, 4),
                "delta_vs_c4": round(delta_vs_c4, 4),
                "verdict_vs_c0": verdict_vs_c0,
                "n_configs_searched": n_done,
            }
            row.update({f"best_{k}": v for k, v in best_cfg.items()})
            results.append(row)

    if results:
        out_path = Path("evaluation/outputs/ga_pre_results.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(results[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✅ Results saved → {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+",
                        default=["heart", "bank", "house_prices"],
                        choices=["heart", "bank", "house_prices"])
    parser.add_argument("--model", nargs="+", default=None)
    args = parser.parse_args()
    run_ga_pre(dataset_ids=args.dataset, model_filter=args.model)
