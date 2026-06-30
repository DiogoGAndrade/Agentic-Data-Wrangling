"""
evaluation/run_automl_comparators.py
=====================================
AutoML Comparators — FLAML, mljar-supervised, Optuna HPO, Bayesian Search

All systems receive the SAME raw data (c0_raw.csv) as C4, evaluated under
the SAME 5-fold CV protocol (RANDOM_STATE=369, StratifiedKFold/KFold).

Systems:
  FLAML        — Microsoft AutoML: searches model family + hyperparams (LightGBM, XGBoost, RF...)
  mljar        — mljar-supervised: ensemble of algorithms with feature engineering
  Optuna       — Bayesian HPO over a fixed RF/LightGBM pipeline (no LLM, no domain knowledge)
                 Directly analogous to GA-PRE but using modern TPE instead of random search.
  BayesSearch  — scikit-optimize BayesSearchCV over RF pipeline (literature comparator:
                 represents classical Bayesian preprocessing search used in several AutoML papers)

Key difference from C4: these systems optimise preprocessing+model automatically.
C4 only does LLM-guided preprocessing; the downstream model is fixed.
This framing is intentional — these AutoML systems are the STRONGEST possible baseline.

Usage (from project root, venv active):
    python -m evaluation.run_automl_comparators --system flaml
    python -m evaluation.run_automl_comparators --system mljar
    python -m evaluation.run_automl_comparators --system optuna
    python -m evaluation.run_automl_comparators --system bayes
    python -m evaluation.run_automl_comparators --system all

Results saved to:
    evaluation/outputs/results_flaml.csv
    evaluation/outputs/results_mljar.csv
    evaluation/outputs/results_optuna.csv
    evaluation/outputs/results_bayes.csv
    evaluation/outputs/automl_comparators_summary.csv  (combined scorecard vs C4)
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import f1_score, r2_score

from engine.config import RANDOM_STATE
from evaluation.run_experiments import DatasetSpec, load_raw

# ---------------------------------------------------------------------------
# Dataset specs — same as run_experiments.py
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

N_FOLDS   = 5
TIME_PER_FOLD_FLAML = 60   # seconds per fold per dataset — adjust up for better results
TIME_FLAML_TOTAL    = 120  # total budget for FLAML per dataset (seconds)
MLJAR_TIME_LIMIT    = 180  # seconds total for mljar per dataset
OPTUNA_TRIALS       = 50   # number of Optuna TPE trials per dataset
BAYES_ITER          = 30   # number of BayesSearchCV iterations per dataset


# ---------------------------------------------------------------------------
# Helper: encode categoricals simply for AutoML systems that need numeric input
# ---------------------------------------------------------------------------
def _basic_encode(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Ordinal-encode all object/category columns. Fit on train, apply to test."""
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.impute import SimpleImputer

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    X_tr = X_train.copy()
    X_te = X_test.copy()

    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_tr[cat_cols] = enc.fit_transform(X_tr[cat_cols].astype(str))
        X_te[cat_cols] = enc.transform(X_te[cat_cols].astype(str))

    # Impute remaining NaNs
    imp = SimpleImputer(strategy="median")
    X_tr = pd.DataFrame(imp.fit_transform(X_tr), columns=X_tr.columns)
    X_te = pd.DataFrame(imp.transform(X_te),     columns=X_te.columns)

    return X_tr, X_te


# ---------------------------------------------------------------------------
# FLAML comparator
# ---------------------------------------------------------------------------
def run_flaml(spec: DatasetSpec, time_budget: int = TIME_FLAML_TOTAL) -> dict:
    """Run FLAML AutoML on spec with 5-fold CV. Returns metrics dict."""
    from flaml import AutoML

    print(f"  [FLAML] {spec.dataset_id} (budget={time_budget}s) ...", flush=True)
    X, y = load_raw(spec)

    cv = (KFold if spec.task_type == "regression" else StratifiedKFold)(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    fold_scores = []
    for fold_i, (tr_idx, te_idx) in enumerate(cv.split(X, y)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        X_tr_enc, X_te_enc = _basic_encode(X_tr, X_te)

        task  = "classification" if spec.task_type == "classification" else "regression"
        metric = "macro_f1" if task == "classification" else "r2"

        automl = AutoML()
        automl.fit(
            X_tr_enc, y_tr,
            task=task,
            metric=metric,
            time_budget=time_budget,
            seed=RANDOM_STATE,
            verbose=0,
        )
        y_pred = automl.predict(X_te_enc)

        if task == "classification":
            score = f1_score(y_te, y_pred, average="macro", zero_division=0)
        else:
            score = r2_score(y_te, y_pred)

        fold_scores.append(score)
        print(f"    fold {fold_i+1}: {score:.4f} (best_model={automl.best_estimator})", flush=True)

    mean_score = float(np.mean(fold_scores))
    std_score  = float(np.std(fold_scores))
    metric_name = "f1_macro" if spec.task_type == "classification" else "r2"

    print(f"  [FLAML] {spec.dataset_id} → {metric_name}={mean_score:.4f} ± {std_score:.4f}", flush=True)
    return {
        "dataset":     spec.dataset_id,
        "system":      "FLAML",
        "task_type":   spec.task_type,
        metric_name:   mean_score,
        f"{metric_name}_std": std_score,
        "n_folds":     N_FOLDS,
        "time_budget": time_budget,
    }


# ---------------------------------------------------------------------------
# mljar-supervised comparator
# ---------------------------------------------------------------------------
def run_mljar(spec: DatasetSpec, time_limit: int = MLJAR_TIME_LIMIT) -> dict:
    """Run mljar AutoML on spec with 5-fold CV. Returns metrics dict."""
    try:
        from supervised.automl import AutoML as MljarAutoML
    except ImportError:
        print(f"  [mljar] NOT INSTALLED — skipping {spec.dataset_id}", flush=True)
        return {}

    import tempfile, os, shutil

    print(f"  [mljar] {spec.dataset_id} (time_limit={time_limit}s) ...", flush=True)
    X, y = load_raw(spec)

    cv = (KFold if spec.task_type == "regression" else StratifiedKFold)(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    fold_scores = []
    for fold_i, (tr_idx, te_idx) in enumerate(cv.split(X, y)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        X_tr_enc, X_te_enc = _basic_encode(X_tr, X_te)

        task    = "binary_classification" if spec.task_type == "classification" and y.nunique() == 2 else \
                  "multiclass_classification" if spec.task_type == "classification" else "regression"
        metric  = "f1" if spec.task_type == "classification" else "r2"

        tmpdir = tempfile.mkdtemp(prefix="mljar_")
        try:
            automl = MljarAutoML(
                mode="Compete",
                total_time_limit=time_limit,
                eval_metric=metric,
                results_path=tmpdir,
                random_state=RANDOM_STATE,
                verbose=0,
            )
            automl.fit(X_tr_enc, y_tr)
            y_pred = automl.predict(X_te_enc)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        if spec.task_type == "classification":
            score = f1_score(y_te, y_pred, average="macro", zero_division=0)
        else:
            score = r2_score(y_te, y_pred)

        fold_scores.append(score)
        print(f"    fold {fold_i+1}: {score:.4f}", flush=True)

    mean_score = float(np.mean(fold_scores))
    std_score  = float(np.std(fold_scores))
    metric_name = "f1_macro" if spec.task_type == "classification" else "r2"

    print(f"  [mljar] {spec.dataset_id} → {metric_name}={mean_score:.4f} ± {std_score:.4f}", flush=True)
    return {
        "dataset":     spec.dataset_id,
        "system":      "mljar",
        "task_type":   spec.task_type,
        metric_name:   mean_score,
        f"{metric_name}_std": std_score,
        "n_folds":     N_FOLDS,
        "time_budget": time_limit,
    }


# ---------------------------------------------------------------------------
# Optuna HPO comparator
# ---------------------------------------------------------------------------
def run_optuna(spec: DatasetSpec, n_trials: int = OPTUNA_TRIALS) -> dict:
    """
    Bayesian HPO (Optuna TPE) over a LightGBM pipeline.
    No LLM, no domain knowledge — pure hyperparameter search.
    Analogous to GA-PRE but with modern TPE sampler instead of random grid.
    Evaluated with same 5-fold CV.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer, make_column_selector as selector
    import lightgbm as lgb

    print(f"  [Optuna] {spec.dataset_id} ({n_trials} trials) ...", flush=True)
    X, y = load_raw(spec)

    cv = (KFold if spec.task_type == "regression" else StratifiedKFold)(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    def make_pipeline(params):
        pre = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), cat_cols),
        ], remainder="passthrough")

        if spec.task_type == "classification":
            model = lgb.LGBMClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                num_leaves=params["num_leaves"],
                random_state=RANDOM_STATE, verbose=-1,
            )
        else:
            model = lgb.LGBMRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                num_leaves=params["num_leaves"],
                random_state=RANDOM_STATE, verbose=-1,
            )
        return Pipeline([("pre", pre), ("model", model)])

    scores_per_trial = []

    def objective(trial):
        params = {
            "n_estimators":  trial.suggest_int("n_estimators", 50, 300),
            "max_depth":     trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":    trial.suggest_int("num_leaves", 15, 127),
        }
        pipe = make_pipeline(params)
        metric = "f1_macro" if spec.task_type == "classification" else "r2"
        scoring = metric if metric == "r2" else "f1_macro"
        fold_scores = []
        for tr_idx, te_idx in cv.split(X, y):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_te)
            if spec.task_type == "classification":
                s = f1_score(y_te, y_pred, average="macro", zero_division=0)
            else:
                s = r2_score(y_te, y_pred)
            fold_scores.append(s)
        mean_s = float(np.mean(fold_scores))
        scores_per_trial.append((params, fold_scores))
        return mean_s

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_trial_idx = np.argmax([np.mean(s) for _, s in scores_per_trial])
    best_folds = scores_per_trial[best_trial_idx][1]
    mean_score = float(np.mean(best_folds))
    std_score  = float(np.std(best_folds))
    metric_name = "f1_macro" if spec.task_type == "classification" else "r2"

    print(f"  [Optuna] {spec.dataset_id} → {metric_name}={mean_score:.4f} ± {std_score:.4f} "
          f"(best params: {study.best_params})", flush=True)
    return {
        "dataset":   spec.dataset_id,
        "system":    "Optuna_LightGBM",
        "task_type": spec.task_type,
        metric_name: mean_score,
        f"{metric_name}_std": std_score,
        "n_folds":   N_FOLDS,
        "n_trials":  n_trials,
        "best_params": str(study.best_params),
    }


# ---------------------------------------------------------------------------
# Bayesian Search (scikit-optimize) comparator
# ---------------------------------------------------------------------------
def run_bayes(spec: DatasetSpec, n_iter: int = BAYES_ITER) -> dict:
    """
    BayesSearchCV (scikit-optimize / skopt) over a Random Forest pipeline.
    Classical Bayesian preprocessing+HPO search — used as reference in several
    AutoML papers. No LLM, no domain knowledge.
    """
    from skopt import BayesSearchCV
    from skopt.space import Integer, Real, Categorical
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer, make_column_selector as selector
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    print(f"  [BayesSearch] {spec.dataset_id} ({n_iter} iters) ...", flush=True)
    X, y = load_raw(spec)
    X_enc, _ = _basic_encode(X, X)  # fit+transform on full dataset for BayesSearchCV

    cv = (KFold if spec.task_type == "regression" else StratifiedKFold)(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    if spec.task_type == "classification":
        model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        scoring = "f1_macro"
    else:
        model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        scoring = "r2"

    search_space = {
        "n_estimators":      Integer(50, 300),
        "max_depth":         Integer(3, 20),
        "min_samples_split": Integer(2, 20),
        "min_samples_leaf":  Integer(1, 10),
        "max_features":      Categorical(["sqrt", "log2", None]),
    }

    opt = BayesSearchCV(
        model, search_space,
        n_iter=n_iter, cv=cv, scoring=scoring,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
        refit=True,
    )
    opt.fit(X_enc, y)

    best_idx  = opt.best_index_
    cv_results = opt.cv_results_
    fold_scores = [cv_results[f"split{i}_test_score"][best_idx] for i in range(N_FOLDS)]
    mean_score = float(np.mean(fold_scores))
    std_score  = float(np.std(fold_scores))
    metric_name = "f1_macro" if spec.task_type == "classification" else "r2"

    print(f"  [BayesSearch] {spec.dataset_id} → {metric_name}={mean_score:.4f} ± {std_score:.4f}", flush=True)
    return {
        "dataset":   spec.dataset_id,
        "system":    "BayesSearch_RF",
        "task_type": spec.task_type,
        metric_name: mean_score,
        f"{metric_name}_std": std_score,
        "n_folds":   N_FOLDS,
        "n_iter":    n_iter,
        "best_params": str(opt.best_params_),
    }


# ---------------------------------------------------------------------------
# Build scorecard vs C4
# ---------------------------------------------------------------------------
def build_scorecard(results: list[dict], master_path: Path) -> pd.DataFrame:
    """Compare AutoML results to C4 best from MASTER using W/T/L criterion."""
    master = pd.read_csv(master_path)

    # C0 baseline per dataset/model (we compare AutoML vs C0 and vs C4 best)
    c0 = master[master["condition"] == "C0_raw"].copy()
    c4 = master[master["condition"].str.startswith("C4_")].copy()

    rows = []
    for r in results:
        if not r:
            continue
        ds = r["dataset"]
        metric = "f1_macro" if r["task_type"] == "classification" else "r2"
        automl_val = r.get(metric, None)
        automl_std = r.get(f"{metric}_std", 0.0)
        if automl_val is None:
            continue

        # C0 for this dataset (average across models)
        c0_ds = c0[c0["dataset"] == ds][metric].mean()
        c0_std_ds = c0[c0["dataset"] == ds][f"{metric}_std"].mean()

        # C4 best for this dataset
        c4_ds = c4[c4["dataset"] == ds][metric].max() if metric in c4.columns else None
        c4_std_ds = c4[c4["dataset"] == ds][f"{metric}_std"].mean() if metric in c4.columns else None

        # W/T/L vs C0
        if c0_ds is not None and not np.isnan(c0_ds):
            delta_vs_c0 = automl_val - c0_ds
            thresh_vs_c0 = c0_std_ds + automl_std
            if delta_vs_c0 > thresh_vs_c0:
                verdict_c0 = "WIN"
            elif delta_vs_c0 < -thresh_vs_c0:
                verdict_c0 = "LOSS"
            else:
                verdict_c0 = "TIE"
        else:
            delta_vs_c0, verdict_c0 = None, "N/A"

        # W/T/L vs C4
        if c4_ds is not None and not np.isnan(c4_ds):
            delta_vs_c4 = automl_val - c4_ds
            thresh_vs_c4 = (c4_std_ds or 0) + automl_std
            if delta_vs_c4 > thresh_vs_c4:
                verdict_c4 = "WIN"
            elif delta_vs_c4 < -thresh_vs_c4:
                verdict_c4 = "LOSS"
            else:
                verdict_c4 = "TIE"
        else:
            delta_vs_c4, verdict_c4 = None, "N/A"

        rows.append({
            "system":         r["system"],
            "dataset":        ds,
            "metric":         metric,
            "automl_val":     round(automl_val, 4),
            "automl_std":     round(automl_std, 4),
            "c0_val":         round(c0_ds, 4) if c0_ds else None,
            "c4_best":        round(c4_ds, 4) if c4_ds else None,
            "delta_vs_c0":    round(delta_vs_c0, 4) if delta_vs_c0 is not None else None,
            "verdict_vs_c0":  verdict_c0,
            "delta_vs_c4":    round(delta_vs_c4, 4) if delta_vs_c4 is not None else None,
            "verdict_vs_c4":  verdict_c4,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["flaml", "mljar", "optuna", "bayes", "all"], default="all")
    parser.add_argument("--datasets", default="all",
                        help="Comma-separated dataset names, or 'all'")
    parser.add_argument("--flaml-budget", type=int, default=TIME_FLAML_TOTAL,
                        help="FLAML time budget in seconds per dataset")
    parser.add_argument("--mljar-limit", type=int, default=MLJAR_TIME_LIMIT,
                        help="mljar time limit in seconds per dataset")
    parser.add_argument("--optuna-trials", type=int, default=OPTUNA_TRIALS,
                        help="Number of Optuna TPE trials per dataset")
    parser.add_argument("--bayes-iter", type=int, default=BAYES_ITER,
                        help="Number of BayesSearchCV iterations per dataset")
    args = parser.parse_args()

    specs = SPECS
    if args.datasets != "all":
        wanted = set(args.datasets.split(","))
        specs = [s for s in SPECS if s.dataset_id in wanted]

    out_dir = Path("evaluation/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    master_path = out_dir / "MASTER_RESULTS_TABLE.csv"

    all_results = []

    # --- FLAML ---
    if args.system in ("flaml", "all"):
        print("\n====== FLAML AutoML Comparator ======", flush=True)
        flaml_results = []
        for spec in specs:
            try:
                r = run_flaml(spec, time_budget=args.flaml_budget)
                flaml_results.append(r)
            except Exception as e:
                print(f"  [FLAML] ERROR on {spec.dataset_id}: {e}", flush=True)
                flaml_results.append({"dataset": spec.dataset_id, "system": "FLAML", "error": str(e)})

        if flaml_results:
            pd.DataFrame(flaml_results).to_csv(out_dir / "results_flaml.csv", index=False)
            print(f"\n  Saved → evaluation/outputs/results_flaml.csv", flush=True)
        all_results.extend(flaml_results)

    # --- Optuna ---
    if args.system in ("optuna", "all"):
        print("\n====== Optuna TPE + LightGBM Comparator ======", flush=True)
        optuna_results = []
        for spec in specs:
            try:
                r = run_optuna(spec, n_trials=args.optuna_trials)
                optuna_results.append(r)
            except Exception as e:
                print(f"  [Optuna] ERROR on {spec.dataset_id}: {e}", flush=True)
                optuna_results.append({"dataset": spec.dataset_id, "system": "Optuna_LightGBM", "error": str(e)})

        if optuna_results:
            pd.DataFrame(optuna_results).to_csv(out_dir / "results_optuna.csv", index=False)
            print(f"\n  Saved → evaluation/outputs/results_optuna.csv", flush=True)
        all_results.extend(optuna_results)

    # --- BayesSearch ---
    if args.system in ("bayes", "all"):
        print("\n====== BayesSearch (scikit-optimize) + RF Comparator ======", flush=True)
        bayes_results = []
        for spec in specs:
            try:
                r = run_bayes(spec, n_iter=args.bayes_iter)
                bayes_results.append(r)
            except Exception as e:
                print(f"  [BayesSearch] ERROR on {spec.dataset_id}: {e}", flush=True)
                bayes_results.append({"dataset": spec.dataset_id, "system": "BayesSearch_RF", "error": str(e)})

        if bayes_results:
            pd.DataFrame(bayes_results).to_csv(out_dir / "results_bayes.csv", index=False)
            print(f"\n  Saved → evaluation/outputs/results_bayes.csv", flush=True)
        all_results.extend(bayes_results)

    # --- mljar ---
    if args.system in ("mljar", "all"):
        print("\n====== mljar-supervised Comparator ======", flush=True)
        mljar_results = []
        for spec in specs:
            try:
                r = run_mljar(spec, time_limit=args.mljar_limit)
                mljar_results.append(r)
            except Exception as e:
                print(f"  [mljar] ERROR on {spec.dataset_id}: {e}", flush=True)
                mljar_results.append({"dataset": spec.dataset_id, "system": "mljar", "error": str(e)})

        if mljar_results:
            pd.DataFrame(mljar_results).to_csv(out_dir / "results_mljar.csv", index=False)
            print(f"\n  Saved → evaluation/outputs/results_mljar.csv", flush=True)
        all_results.extend(mljar_results)

    # --- Scorecard ---
    valid = [r for r in all_results if r and "error" not in r]
    if valid and master_path.exists():
        scorecard = build_scorecard(valid, master_path)
        scorecard.to_csv(out_dir / "automl_comparators_summary.csv", index=False)
        print("\n====== SCORECARD vs C4 ======")
        print(scorecard.to_string(index=False))
        print(f"\n  Saved → evaluation/outputs/automl_comparators_summary.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
