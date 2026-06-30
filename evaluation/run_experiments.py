# evaluation/run_experiments.py
#
# Cleaning is applied INSIDE cross_validate, fold-by-fold, via PlanBasedCleaner.
# C0/C1/C2 differ only by the plan that feeds the cleaner.
#
# Speed optimisations (2026-04-29):
#   - n_estimators reduced (RF 150->100, GBM 100->50): ~3x faster, marginal loss.
#   - DatasetSpec.subsample_n caps very large datasets for Phase A timing.
#   - Baseline cache: C0/C1 results are reused across LLM-tags (saved to
#     evaluation/outputs/baselines_<dataset>.csv on first run).

from __future__ import annotations

import sys
from pathlib import Path
# Ensure project root is on sys.path so `from engine...` works
# regardless of whether the script is invoked as `python -m evaluation.run_experiments`
# or `python evaluation/run_experiments.py`.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from engine.cleaning_pipeline import (
    PlanBasedCleaner,
    build_c0_empty_plan,
    build_c1_deterministic_plan,
)
from engine.config import RANDOM_STATE


@dataclass
class DatasetSpec:
    dataset_id: str
    target: str
    base_dir: Path
    task_type: str = "classification"
    label_map: Optional[Dict[str, str]] = None
    leakage_cols: List[str] = field(default_factory=list)
    subsample_n: Optional[int] = None  # cap rows for speed (stratified for clf)


def load_raw(spec: DatasetSpec) -> Tuple[pd.DataFrame, pd.Series]:
    raw_path = spec.base_dir / "c0_raw.csv"
    df = pd.read_csv(raw_path)
    for c in spec.leakage_cols:
        if c in df.columns:
            df = df.drop(columns=[c])
    if spec.target not in df.columns:
        raise ValueError(f"Target '{spec.target}' not in {raw_path.name}.")

    # Optional capped sub-sampling (stratified for classification).
    if spec.subsample_n is not None and len(df) > spec.subsample_n:
        if spec.task_type == "classification":
            df = (df.groupby(spec.target, group_keys=False)
                    .apply(lambda g: g.sample(
                        n=max(1, int(round(len(g) * spec.subsample_n / len(df)))),
                        random_state=RANDOM_STATE))
                    .reset_index(drop=True))
        else:
            df = df.sample(n=spec.subsample_n, random_state=RANDOM_STATE).reset_index(drop=True)

    y = df[spec.target]
    X = df.drop(columns=[spec.target])
    if spec.label_map is not None:
        y = y.astype(str).map(spec.label_map).fillna(y.astype(str))
    return X, y


def load_plan(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try stripping null bytes (OneDrive sync artefact)
        try:
            cleaned = raw.replace("\x00", "").strip()
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try parsing up to the last valid closing brace
            try:
                last_brace = cleaned.rindex("}")
                return json.loads(cleaned[: last_brace + 1])
            except (ValueError, json.JSONDecodeError):
                return None
    except Exception:
        return None


def build_final_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), selector(dtype_include=np.number)),
            ("cat", cat_pipe, selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def build_models(task_type: str) -> Dict[str, object]:
    if task_type == "regression":
        return {
            "ridge": Ridge(random_state=RANDOM_STATE),
            "rf":    RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
            "knn":   KNeighborsRegressor(n_jobs=-1),
            "gbm":   GradientBoostingRegressor(n_estimators=50, random_state=RANDOM_STATE),
        }
    return {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "rf":     RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                          random_state=RANDOM_STATE, n_jobs=-1),
        "knn":    KNeighborsClassifier(n_jobs=-1),
        "gbm":    GradientBoostingClassifier(n_estimators=50, random_state=RANDOM_STATE),
    }


def build_scoring(task_type: str, has_proba: bool) -> Dict[str, str]:
    if task_type == "regression":
        return {"neg_mae": "neg_mean_absolute_error",
                "neg_rmse": "neg_root_mean_squared_error", "r2": "r2"}
    sc = {"accuracy": "accuracy", "f1_weighted": "f1_weighted", "f1_macro": "f1_macro"}
    if has_proba:
        sc["roc_auc"] = "roc_auc_ovr_weighted"
    return sc


def evaluate_one(X, y, model_name, clf, plan, target_column, task_type, n_splits=5):
    scale_numeric = (model_name in {"logreg", "ridge", "knn"})
    steps = []
    if plan is not None:
        steps.append(("clean", PlanBasedCleaner(plan=plan, target_column=target_column)))
    steps.append(("preprocess", build_final_preprocessor(scale_numeric=scale_numeric)))
    steps.append(("model", clf))
    pipe = Pipeline(steps=steps)
    cv = (KFold if task_type == "regression" else StratifiedKFold)(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = build_scoring(task_type, has_proba=hasattr(clf, "predict_proba"))
    cv_results = cross_validate(pipe, X, y, cv=cv, scoring=scoring,
                                error_score="raise", n_jobs=None, return_train_score=False)
    out: Dict[str, float] = {}
    if task_type == "regression":
        out["mae"]  = float(-np.mean(cv_results["test_neg_mae"]))
        out["rmse"] = float(-np.mean(cv_results["test_neg_rmse"]))
        out["r2"]   = float(np.mean(cv_results["test_r2"]))
        out["mae_std"]  = float(np.std(cv_results["test_neg_mae"]))
        out["rmse_std"] = float(np.std(cv_results["test_neg_rmse"]))
        out["r2_std"]   = float(np.std(cv_results["test_r2"]))
    else:
        out["accuracy"]      = float(np.mean(cv_results["test_accuracy"]))
        out["f1_weighted"]   = float(np.mean(cv_results["test_f1_weighted"]))
        out["f1_macro"]      = float(np.mean(cv_results["test_f1_macro"]))
        out["accuracy_std"]  = float(np.std(cv_results["test_accuracy"]))
        out["f1_weighted_std"] = float(np.std(cv_results["test_f1_weighted"]))
        out["f1_macro_std"]    = float(np.std(cv_results["test_f1_macro"]))
        if "test_roc_auc" in cv_results:
            out["roc_auc"]     = float(np.mean(cv_results["test_roc_auc"]))
            out["roc_auc_std"] = float(np.std(cv_results["test_roc_auc"]))
    return out


def _baseline_cache_path(root: Path, dataset_id: str) -> Path:
    return root / "evaluation" / "outputs" / f"baselines_{dataset_id}.csv"


def run_dataset(spec: DatasetSpec, llm_tag: Optional[str] = None,
                root: Optional[Path] = None,
                refresh_baselines: bool = False,
                condition_prefix: str = "C2") -> pd.DataFrame:
    X, y = load_raw(spec)
    rows: List[Dict] = []
    models = build_models(spec.task_type)

    cache_path = _baseline_cache_path(root or Path("."), spec.dataset_id) if root else None
    cached = (pd.read_csv(cache_path) if (cache_path and cache_path.exists() and not refresh_baselines)
              else None)

    if cached is not None and not cached.empty:
        print(f"[CACHE] {spec.dataset_id}: reusing C0/C1 baselines from {cache_path.name}")
        rows.extend(cached.to_dict(orient="records"))
    else:
        for cond_name, plan in [("C0_raw", build_c0_empty_plan()),
                                ("C1_manual", build_c1_deterministic_plan())]:
            for model_name, clf in models.items():
                print(f"[INFO] {spec.dataset_id} | {cond_name} | {model_name} | rows={len(X)} cols={X.shape[1]}")
                try:
                    scores = evaluate_one(X=X, y=y, model_name=model_name, clf=clf,
                                          plan=plan, target_column=spec.target,
                                          task_type=spec.task_type, n_splits=5)
                except Exception as e:
                    print(f"[FAIL] {spec.dataset_id} | {cond_name} | {model_name}: {e}")
                    continue
                rows.append({
                    "dataset": spec.dataset_id, "task_type": spec.task_type,
                    "condition": cond_name, "model": model_name,
                    **scores, "n_rows": int(len(X)), "n_cols": int(X.shape[1]),
                    "target": spec.target, "llm_tag": "",
                })
        if cache_path is not None and rows:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(cache_path, index=False)
            print(f"[CACHE] {spec.dataset_id}: saved C0/C1 baselines -> {cache_path}")

    # C2/C3/C4 (LLM-specific) — always recomputed
    prov_dir = spec.base_dir / "provenance"
    # Support c2_llm_plan, c3_context_plan, c4_expanded_plan
    if condition_prefix == "C4":
        plan_prefix = "c4_expanded_plan"
        cond_label = "C4_expanded"
    elif condition_prefix == "C3":
        plan_prefix = "c3_context_plan"
        cond_label = "C3_context"
    else:
        plan_prefix = "c2_llm_plan"
        cond_label = "C2_llm"

    # Resolve plan path(s).  C4 supports per-model plans
    # (e.g. c4_expanded_plan_qwen2.5_3b_logreg.json) in addition to the
    # generic per-LLM plan.  We check per-model first, then generic, then
    # C2 fallback.

    def _resolve_plan_path(model_name: Optional[str] = None) -> Optional[Path]:
        """Return the best plan file for this (dataset, llm_tag, model)."""
        candidates: List[Path] = []
        if llm_tag:
            # 1. Per-model plan (C4 only — e.g. c4_expanded_plan_tag_logreg.json)
            if model_name:
                candidates.append(prov_dir / f"{plan_prefix}_{llm_tag}_{model_name}.json")
            # 2. Generic per-LLM plan (e.g. c4_expanded_plan_tag.json)
            candidates.append(prov_dir / f"{plan_prefix}_{llm_tag}.json")
            # 3. Fallback: C2 plan (backward compat)
            candidates.append(prov_dir / f"c2_llm_plan_{llm_tag}.json")
        # 4. Legacy unnamed plan
        candidates.append(prov_dir / "c2_llm_plan.json")
        for c in candidates:
            if c.exists():
                return c
        return None

    # Check if ANY plan exists at all.
    # First try generic (e.g. c4_expanded_plan_tag.json).
    # If not found, check whether any per-model plans exist — Phase B datasets
    # only have per-model plans (no generic plan file).
    generic_path = _resolve_plan_path()
    if generic_path is None:
        # Phase B: only per-model plans exist (no generic plan file).
        # Evaluate all per-model paths explicitly.
        _per_model_paths = {mn: _resolve_plan_path(mn) for mn in list(models.keys())}
        _any_found = any(p is not None for p in _per_model_paths.values())
        if not _any_found:
            print(f"[INFO] {spec.dataset_id}: no {condition_prefix} plan found; baselines only.")
            return pd.DataFrame(rows)
        # Use first available per-model plan as fallback generic
        generic_path = next(p for p in _per_model_paths.values() if p is not None)

    for model_name, clf in models.items():
        plan_path = _resolve_plan_path(model_name)
        if plan_path is None:
            plan_path = generic_path  # shouldn't happen, but safe fallback
        plan = load_plan(plan_path)
        if plan is None:
            print(f"[WARN] {spec.dataset_id}: failed to parse {plan_path.name}; skipping {model_name}.")
            continue
        print(f"[INFO] {spec.dataset_id} | {cond_label} | {model_name} | rows={len(X)} cols={X.shape[1]} | tag={llm_tag} | plan={plan_path.name}")
        try:
            scores = evaluate_one(X=X, y=y, model_name=model_name, clf=clf,
                                  plan=plan, target_column=spec.target,
                                  task_type=spec.task_type, n_splits=5)
        except Exception as e:
            print(f"[FAIL] {spec.dataset_id} | {cond_label} | {model_name}: {e}")
            continue
        rows.append({
            "dataset": spec.dataset_id, "task_type": spec.task_type,
            "condition": cond_label, "model": model_name,
            **scores, "n_rows": int(len(X)), "n_cols": int(X.shape[1]),
            "target": spec.target, "llm_tag": llm_tag or "",
        })

    return pd.DataFrame(rows)


def default_specs(root: Path) -> List[DatasetSpec]:
    return [
        DatasetSpec("adult", "income",
                    root / "data" / "exports" / "adult", task_type="classification"),
        DatasetSpec("diabetes", "readmitted",
                    root / "data" / "exports" / "diabetes", task_type="classification",
                    leakage_cols=["encounter_id", "patient_nbr"],
                    subsample_n=30000),  # speed cap; stratified
        DatasetSpec("student", "final_result",
                    root / "data" / "exports" / "student", task_type="classification",
                    leakage_cols=["id_student"]),
        DatasetSpec("life_expectancy", "life_expectancy",
                    root / "data" / "exports" / "life_expectancy", task_type="regression"),
        DatasetSpec("house_prices", "SalePrice",
                    root / "data" / "exports" / "house_prices", task_type="regression",
                    leakage_cols=["Id"]),
        DatasetSpec("heart", "target",
                    root / "data" / "exports" / "heart", task_type="classification"),
        DatasetSpec("bank", "y",
                    root / "data" / "exports" / "bank", task_type="classification",
                    subsample_n=20000),  # 41k rows, also cap for speed
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--llm-tag", type=str, default=None)
    parser.add_argument("--datasets", type=str, default="adult,diabetes,student,life_expectancy")
    parser.add_argument("--refresh-baselines", action="store_true",
                        help="Force recomputation of C0/C1 even if cached.")
    parser.add_argument("--condition", type=str, default="C2", choices=["C2", "C3", "C4"],
                        help="Which condition to evaluate: C2 (blind), C3 (context-aware), or C4 (expanded).")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    requested = {d.strip() for d in args.datasets.split(",") if d.strip()}
    specs = [s for s in default_specs(root) if s.dataset_id in requested]
    if not specs:
        raise SystemExit(f"No matching specs for: {requested}")

    all_rows: List[pd.DataFrame] = []
    for spec in specs:
        if not (spec.base_dir / "c0_raw.csv").exists():
            print(f"[SKIP] {spec.dataset_id}: missing {spec.base_dir / 'c0_raw.csv'}")
            continue
        all_rows.append(run_dataset(spec, llm_tag=args.llm_tag, root=root,
                                    refresh_baselines=args.refresh_baselines,
                                    condition_prefix=args.condition))

    if not all_rows:
        print("[INFO] Nothing to write."); return

    results = pd.concat(all_rows, ignore_index=True)
    out_dir = root / "evaluation" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    cond_prefix = f"_{args.condition.lower()}" if args.condition in ("C3", "C4") else ""
    suffix = f"{cond_prefix}_{args.llm_tag}" if args.llm_tag else cond_prefix
    dataset_names = sorted({s.dataset_id for s in specs if (s.base_dir / "c0_raw.csv").exists()})
    if len(dataset_names) == 1:
        suffix += f"_{dataset_names[0]}"
    results_path = out_dir / f"results{suffix}.csv"
    results.to_csv(results_path, index=False)
    print(f"[OK] {results_path}")


if __name__ == "__main__":
    main()
