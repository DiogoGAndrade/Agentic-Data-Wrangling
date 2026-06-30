"""
evaluation/apply_cloud_llm_plan.py
====================================
Applies a cleaning plan from a cloud LLM (ChatGPT, Gemini, Claude, Copilot)
to a dataset using the same 5-fold CV protocol as C4.

The cloud LLM generates a JSON plan via the browser (you copy-paste).
This script applies that plan and produces a result row for the MASTER table.

Usage:
    # After pasting the LLM's JSON into evaluation/cloud_llm_comparator/plan_<llm>_<dataset>.json
    python -m evaluation.apply_cloud_llm_plan \
        --llm chatgpt \
        --dataset adult

    # Run all datasets for one LLM
    python -m evaluation.apply_cloud_llm_plan --llm gemini --dataset all

    # Run all LLMs and all datasets
    python -m evaluation.apply_cloud_llm_plan --llm all --dataset all

Results appended to:
    evaluation/outputs/results_cloud_llm_<llm>.csv
    (run python -m evaluation.consolidate_results afterwards to add to MASTER)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from engine.config import RANDOM_STATE
from engine.cleaning_pipeline import PlanBasedCleaner
from evaluation.run_experiments import DatasetSpec, load_raw

PLAN_DIR = Path("evaluation/cloud_llm_comparator")
OUT_DIR  = Path("evaluation/outputs")

SPECS = {
    "adult":           DatasetSpec("adult",           "income",          Path("data/exports/adult"),           "classification"),
    "diabetes":        DatasetSpec("diabetes",        "readmitted",      Path("data/exports/diabetes"),         "classification", leakage_cols=["encounter_id","patient_nbr"]),
    "student":         DatasetSpec("student",         "final_result",    Path("data/exports/student"),          "classification"),
    "life_expectancy": DatasetSpec("life_expectancy", "life_expectancy", Path("data/exports/life_expectancy"),  "regression"),
    "heart":           DatasetSpec("heart",           "target",          Path("data/exports/heart"),            "classification"),
    "bank":            DatasetSpec("bank",            "y",               Path("data/exports/bank"),             "classification"),
    "house_prices":    DatasetSpec("house_prices",    "SalePrice",       Path("data/exports/house_prices"),     "regression", leakage_cols=["Id"]),
}

ML_MODELS = {
    "regression":     {"ridge": Ridge(random_state=RANDOM_STATE),
                       "rf":    RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
                       "knn":   KNeighborsRegressor(n_neighbors=5),
                       "gbm":   GradientBoostingRegressor(n_estimators=50, random_state=RANDOM_STATE)},
    "classification": {"logreg": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
                       "rf":     RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
                       "knn":    KNeighborsClassifier(n_neighbors=5),
                       "gbm":    GradientBoostingClassifier(n_estimators=50, random_state=RANDOM_STATE)},
}


def build_preprocessor(scale: bool) -> ColumnTransformer:
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), selector(dtype_exclude="object")),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         selector(dtype_include="object")),
    ], remainder="passthrough")


def run_one(llm_tag: str, dataset_id: str) -> list[dict]:
    plan_path = PLAN_DIR / f"plan_{llm_tag}_{dataset_id}.json"
    if not plan_path.exists():
        print(f"  [SKIP] Plan not found: {plan_path}")
        return []

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    spec = SPECS[dataset_id]
    X, y = load_raw(spec)

    models = ML_MODELS[spec.task_type]
    scoring = ({"neg_rmse": "neg_root_mean_squared_error", "r2": "r2"}
               if spec.task_type == "regression"
               else {"accuracy": "accuracy", "f1_weighted": "f1_weighted", "f1_macro": "f1_macro"})

    cv = (KFold if spec.task_type == "regression" else StratifiedKFold)(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for model_name, model in models.items():
        scale = model_name in ("logreg", "knn", "ridge")
        pipe = Pipeline([
            ("cleaner",       PlanBasedCleaner(plan=plan)),
            ("preprocessor",  build_preprocessor(scale=scale)),
            ("model",         model),
        ])
        res = cross_validate(pipe, X, y, cv=cv, scoring=scoring,
                             return_train_score=False, error_score="raise")
        row = {
            "dataset":    dataset_id,
            "task_type":  spec.task_type,
            "condition":  f"C6_{llm_tag}",
            "model":      model_name,
            "llm_tag":    llm_tag,
            "n_rows":     len(X),
            "n_cols":     X.shape[1],
            "target":     spec.target,
        }
        if spec.task_type == "regression":
            row["r2"]       = float(np.mean(res["test_r2"]))
            row["r2_std"]   = float(np.std(res["test_r2"]))
            row["rmse"]     = float(-np.mean(res["test_neg_rmse"]))
            row["rmse_std"] = float(np.std(res["test_neg_rmse"]))
        else:
            row["accuracy"]       = float(np.mean(res["test_accuracy"]))
            row["f1_weighted"]    = float(np.mean(res["test_f1_weighted"]))
            row["f1_macro"]       = float(np.mean(res["test_f1_macro"]))
            row["f1_macro_std"]   = float(np.std(res["test_f1_macro"]))
        print(f"  [{llm_tag}] {dataset_id}/{model_name}: "
              f"{'f1_macro' if spec.task_type=='classification' else 'r2'}="
              f"{row.get('f1_macro', row.get('r2', '?')):.4f}", flush=True)
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm",     default="all",
                        help="LLM tag (e.g. chatgpt, gemini, claude, copilot) or 'all'")
    parser.add_argument("--dataset", default="all",
                        help="Dataset name or 'all'")
    args = parser.parse_args()

    # Auto-detect which plans exist
    all_plans = list(PLAN_DIR.glob("plan_*.json"))
    if not all_plans:
        print("No plans found in evaluation/cloud_llm_comparator/")
        print("Expected filenames: plan_<llm>_<dataset>.json")
        print("e.g.: plan_chatgpt_adult.json, plan_gemini_heart.json")
        return

    available = {}
    for p in all_plans:
        parts = p.stem.split("_", 2)  # plan, llm, dataset
        if len(parts) == 3:
            llm, ds = parts[1], parts[2]
            available.setdefault(llm, []).append(ds)

    llms    = list(available.keys()) if args.llm == "all" else [args.llm]
    datasets = list(SPECS.keys())    if args.dataset == "all" else [args.dataset]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for llm in llms:
        print(f"\n====== Cloud LLM: {llm} ======", flush=True)
        llm_rows = []
        for ds in datasets:
            rows = run_one(llm, ds)
            llm_rows.extend(rows)
            all_rows.extend(rows)

        if llm_rows:
            out_path = OUT_DIR / f"results_cloud_llm_{llm}.csv"
            pd.DataFrame(llm_rows).to_csv(out_path, index=False)
            print(f"  Saved → {out_path}")

    if all_rows:
        combined = OUT_DIR / "results_cloud_llm_all.csv"
        pd.DataFrame(all_rows).to_csv(combined, index=False)
        print(f"\nCombined → {combined}")
        print("\nNext step: run  python -m evaluation.consolidate_results  to add to MASTER")


if __name__ == "__main__":
    main()
