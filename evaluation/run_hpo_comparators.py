"""
evaluation/run_hpo_comparators.py
===================================
HPO Comparators — Optuna (TPE + LightGBM) and BayesSearch (skopt + RF)

These represent classical Bayesian/TPE hyperparameter optimisation —
no LLM, no domain knowledge, pure search over model hyperparameters.
They sit between GA-PRE (random grid) and FLAML (full AutoML) in
the comparator spectrum.

Literature relevance:
  - Optuna TPE: Akiba et al. (2019) — standard modern HPO baseline
  - BayesSearchCV: skopt — classical Bayesian opt used in many AutoML papers
    (Bergstra & Bengio 2012, Snoek et al. 2012)

Evaluated with the SAME 5-fold CV as C4 (RANDOM_STATE=369).

Usage:
    python -m evaluation.run_hpo_comparators --system optuna
    python -m evaluation.run_hpo_comparators --system bayes
    python -m evaluation.run_hpo_comparators --system all
    python -m evaluation.run_hpo_comparators --system all --datasets heart,bank

Results saved to:
    evaluation/outputs/results_optuna.csv
    evaluation/outputs/results_bayes.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold

from engine.config import RANDOM_STATE
from evaluation.run_experiments import DatasetSpec, load_raw

# ---------------------------------------------------------------------------
# Dataset specs — identical to run_experiments
# ---------------------------------------------------------------------------
SPECS = [
    DatasetSpec("adult",           "income",          Path("data/exports/adult"),           "classification", subsample_n=20_000),
    DatasetSpec("diabetes",        "readmitted",      Path("data/exports/diabetes"),         "classification", subsample_n=10_000, leakage_cols=["encounter_id", "patient_nbr"]),
    DatasetSpec("student",         "final_result",    Path("data/exports/student"),          "classification"),
    DatasetSpec("life_expectancy", "life_expectancy", Path("data/exports/life_expectancy"),  "regression"),
    DatasetSpec("heart",           "target",          Path("data/exports/heart"),            "classification"),
    DatasetSpec("bank",            "y",               Path("data/exports/bank"),             "classification", subsample_n=10_000),
    DatasetSpec("house_prices",    "SalePrice",       Path("data/exports/house_prices"),     "regression", leakage_cols=["Id"]),
]

N_FOLDS        = 5
OPTUNA_TRIALS  = 50
BAYES_ITER     = 30


def _encode(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Ordinal-encode categoricals + impute. Fit on train only."""
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.impute import SimpleImputer

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    X_tr = X_train.copy()
    X_te = X_test.copy()
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_tr[cat_cols] = enc.fit_transform(X_tr[cat_cols].astype(str))
        X_te[cat_cols] = enc.transform(X_te[cat_cols].astype(str))
    imp = SimpleImputer(strategy="median")
    X_tr = pd.DataFrame(imp.fit_transform(X_tr), columns=X_tr.columns)
    X_te = pd.DataFrame(imp.transform(X_te),     columns=X_te.columns)
    return X_tr, X_te


def _encode_full(X: pd.DataFrame):
    """Encode full dataset (for BayesSearchCV which handles CV internally)."""
    Xtr, _ = _encode(X, X)
    return Xtr


# ---------------------------------------------------------------------------
# Optuna + LightGBM
# ---------------------------------------------------------------------------
def run_optuna(spec: DatasetSpec, n_trials: int = OPTUNA_TRIALS) -> dict:
    """
    Bayesian HPO (Optuna TPE) over a LightGBM pipeline.
    Searches: n_estimators, max_depth, learning_rate, num_leaves.
    Evaluated fold-by-fold with same 5-fold CV as C4.
    """
    import optuna
    import lightgbm as lgb
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"  [Optuna] {spec.dataset_id} ({n_trials} trials) ...", flush=True)
    X, y = load_raw(spec)

    cv_splitter = (KFold if spec.task_type == "regression" else StratifiedKFold)(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    folds = list(cv_splitter.split(X, y))

    def objective(trial):
        params = dict(
            n_estimators  = trial.suggest_int("n_estimators", 50, 400),
            max_depth      = trial.suggest_int("max_depth", 3, 12),
            learning_rate  = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves     = trial.suggest_int("num_leaves", 15, 127),
            min_child_samples = trial.suggest_int("min_child_samples", 5, 50),
        )
        fold_scores = []
        for tr_idx, te_idx in folds:
            X_tr, X_te = _encode(X.iloc[tr_idx], X.iloc[te_idx])
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

            Model = lgb.LGBMClassifier if spec.task_type == "classification" else lgb.LGBMRegressor
            m = Model(**params, random_state=RANDOM_STATE, verbose=-1)
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_te)

            if spec.task_type == "classification":
                fold_scores.append(f1_score(y_te, y_pred, average="macro", zero_division=0))
            else:
                fold_scores.append(r2_score(y_te, y_pred))
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Re-evaluate best params to get per-fold scores for std
    best_params = study.best_params
    fold_scores = []
    for tr_idx, te_idx in folds:
        X_tr, X_te = _encode(X.iloc[tr_idx], X.iloc[te_idx])
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        Model = lgb.LGBMClassifier if spec.task_type == "classification" else lgb.LGBMRegressor
        m = Model(**best_params, random_state=RANDOM_STATE, verbose=-1)
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        if spec.task_type == "classification":
            fold_scores.append(f1_score(y_te, y_pred, average="macro", zero_division=0))
        else:
            fold_scores.append(r2_score(y_te, y_pred))

    mean_s = float(np.mean(fold_scores))
    std_s  = float(np.std(fold_scores))
    metric = "f1_macro" if spec.task_type == "classification" else "r2"
    print(f"  [Optuna] {spec.dataset_id} → {metric}={mean_s:.4f} ± {std_s:.4f}  "
          f"best={best_params}", flush=True)
    return {
        "dataset": spec.dataset_id, "system": "Optuna_LightGBM",
        "task_type": spec.task_type,
        metric: mean_s, f"{metric}_std": std_s,
        "n_folds": N_FOLDS, "n_trials": n_trials,
        "best_params": str(best_params),
    }


# ---------------------------------------------------------------------------
# BayesSearch (scikit-optimize) + Random Forest
# ---------------------------------------------------------------------------
def run_bayes(spec: DatasetSpec, n_iter: int = BAYES_ITER) -> dict:
    """
    BayesSearchCV (skopt) over a Random Forest.
    Classical Bayesian preprocessing+HPO — reference method from AutoML literature.
    """
    from skopt import BayesSearchCV
    from skopt.space import Integer, Categorical
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    print(f"  [BayesSearch] {spec.dataset_id} ({n_iter} iters) ...", flush=True)
    X, y = load_raw(spec)
    X_enc = _encode_full(X)

    cv_splitter = (KFold if spec.task_type == "regression" else StratifiedKFold)(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    if spec.task_type == "classification":
        base_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        scoring = "f1_macro"
    else:
        base_model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        scoring = "r2"

    search = BayesSearchCV(
        base_model,
        {
            "n_estimators":      Integer(50, 400),
            "max_depth":         Integer(3, 25),
            "min_samples_split": Integer(2, 20),
            "min_samples_leaf":  Integer(1, 10),
            "max_features":      Categorical(["sqrt", "log2"]),
        },
        n_iter=n_iter,
        cv=cv_splitter,
        scoring=scoring,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    search.fit(X_enc, y)

    best_i = search.best_index_
    cv_res = search.cv_results_
    fold_scores = [cv_res[f"split{i}_test_score"][best_i] for i in range(N_FOLDS)]
    mean_s = float(np.mean(fold_scores))
    std_s  = float(np.std(fold_scores))
    metric = "f1_macro" if spec.task_type == "classification" else "r2"

    print(f"  [BayesSearch] {spec.dataset_id} → {metric}={mean_s:.4f} ± {std_s:.4f}  "
          f"best={search.best_params_}", flush=True)
    return {
        "dataset": spec.dataset_id, "system": "BayesSearch_RF",
        "task_type": spec.task_type,
        metric: mean_s, f"{metric}_std": std_s,
        "n_folds": N_FOLDS, "n_iter": n_iter,
        "best_params": str(search.best_params_),
    }


# ---------------------------------------------------------------------------
# Build W/T/L scorecard vs C4
# ---------------------------------------------------------------------------
def build_scorecard(results: list, master_path: Path) -> pd.DataFrame:
    master = pd.read_csv(master_path)
    c0 = master[master["condition"] == "C0_raw"]
    c4 = master[master["condition"].str.startswith("C4_")]
    rows = []
    for r in results:
        if not r or "error" in r:
            continue
        ds     = r["dataset"]
        metric = "f1_macro" if r["task_type"] == "classification" else "r2"
        val    = r.get(metric)
        std    = r.get(f"{metric}_std", 0.0)
        if val is None:
            continue
        c0_val = c0[c0["dataset"] == ds][metric].mean()
        c0_std = c0[c0["dataset"] == ds].get(f"{metric}_std", pd.Series([0.0])).mean()
        c4_val = c4[c4["dataset"] == ds][metric].max() if metric in c4.columns else None

        def verdict(delta, thresh):
            if delta > thresh:  return "WIN"
            if delta < -thresh: return "LOSS"
            return "TIE"

        rows.append({
            "system":        r["system"],
            "dataset":       ds,
            "metric":        metric,
            "value":         round(val, 4),
            "std":           round(std, 4),
            "c0":            round(c0_val, 4) if not np.isnan(c0_val) else None,
            "c4_best":       round(c4_val, 4) if c4_val is not None else None,
            "delta_vs_c0":   round(val - c0_val, 4) if not np.isnan(c0_val) else None,
            "verdict_vs_c0": verdict(val - c0_val, c0_std + std) if not np.isnan(c0_val) else "N/A",
            "delta_vs_c4":   round(val - c4_val, 4) if c4_val is not None else None,
            "verdict_vs_c4": verdict(val - c4_val, std) if c4_val is not None else "N/A",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="HPO comparators: Optuna + BayesSearch")
    parser.add_argument("--system",       choices=["optuna", "bayes", "all"], default="all")
    parser.add_argument("--datasets",     default="all")
    parser.add_argument("--optuna-trials", type=int, default=OPTUNA_TRIALS)
    parser.add_argument("--bayes-iter",   type=int, default=BAYES_ITER)
    args = parser.parse_args()

    specs = SPECS
    if args.datasets != "all":
        wanted = set(args.datasets.split(","))
        specs  = [s for s in SPECS if s.dataset_id in wanted]

    out = Path("evaluation/outputs")
    out.mkdir(parents=True, exist_ok=True)
    master = out / "MASTER_RESULTS_TABLE.csv"
    all_results = []

    if args.system in ("optuna", "all"):
        print("\n====== Optuna TPE + LightGBM ======", flush=True)
        res = []
        for s in specs:
            try:
                res.append(run_optuna(s, n_trials=args.optuna_trials))
            except Exception as e:
                print(f"  ERROR {s.dataset_id}: {e}", flush=True)
                res.append({"dataset": s.dataset_id, "system": "Optuna_LightGBM", "error": str(e)})
        pd.DataFrame(res).to_csv(out / "results_optuna.csv", index=False)
        print(f"  Saved → evaluation/outputs/results_optuna.csv")
        all_results.extend(res)

    if args.system in ("bayes", "all"):
        print("\n====== BayesSearch (skopt) + RF ======", flush=True)
        res = []
        for s in specs:
            try:
                res.append(run_bayes(s, n_iter=args.bayes_iter))
            except Exception as e:
                print(f"  ERROR {s.dataset_id}: {e}", flush=True)
                res.append({"dataset": s.dataset_id, "system": "BayesSearch_RF", "error": str(e)})
        pd.DataFrame(res).to_csv(out / "results_bayes.csv", index=False)
        print(f"  Saved → evaluation/outputs/results_bayes.csv")
        all_results.extend(res)

    valid = [r for r in all_results if r and "error" not in r]
    if valid and master.exists():
        sc = build_scorecard(valid, master)
        sc.to_csv(out / "hpo_comparators_summary.csv", index=False)
        print("\n====== SCORECARD vs C4 ======")
        print(sc.to_string(index=False))
        print(f"\n  Saved → evaluation/outputs/hpo_comparators_summary.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
