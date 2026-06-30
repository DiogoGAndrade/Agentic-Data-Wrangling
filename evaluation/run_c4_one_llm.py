"""
Run C4 evaluation for a single LLM tag. Called as:
  python evaluation/run_c4_one_llm.py <llm_tag>
e.g.:
  python evaluation/run_c4_one_llm.py qwen2_5_3b
"""
import copy, json, sys, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from engine.cleaning_pipeline import PlanBasedCleaner
from engine.config import RANDOM_STATE
from evaluation.enforce_c4_v3 import enforce_plan, compute_cardinality
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score, r2_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

EXPORTS = ROOT / "data" / "exports"
OUT = ROOT / "evaluation" / "outputs"
N_FOLDS = 5

PHASE_C = {
    "platform": {
        "target": "purchased", "task": "classification",
        "user_context": {
            "downstream_model": "RandomForest",
            "column_semantics": {
                "age": "numeric", "income": "numeric",
                "days_on_platform": "numeric",
                "gender": "categorical", "city": "categorical",
            },
            "redundant_features": [], "leakage_cols": [],
        },
    },
    "support2_clf": {
        "target": "hospdead", "task": "classification",
        "user_context": {
            "downstream_model": "RandomForest",
            "column_semantics": {
                "age": "numeric", "meanbp": "numeric", "hrt": "numeric",
                "resp": "numeric", "temp": "numeric",
                "sps": "numeric", "aps": "numeric",
                "scoma": "numeric", "totcst": "numeric", "totmcst": "numeric",
                "avtisst": "numeric", "surv2m": "numeric", "surv6m": "numeric",
                "hday": "numeric", "prg2m": "numeric", "prg6m": "numeric",
                "wblc": "numeric", "pafi": "numeric", "alb": "numeric",
                "bili": "numeric", "crea": "numeric", "sod": "numeric",
                "ph": "numeric", "glucose": "numeric", "bun": "numeric",
                "urine": "numeric", "adlp": "numeric", "adls": "numeric",
                "adlsc": "numeric", "edu": "numeric", "num.co": "numeric",
                "dnrday": "numeric",
            },
            "redundant_features": [], "leakage_cols": [],
        },
    },
    "support2_reg": {
        "target": "log_charges", "task": "regression",
        "user_context": {
            "downstream_model": "RandomForest",
            "column_semantics": {
                "age": "numeric", "meanbp": "numeric", "hrt": "numeric",
                "resp": "numeric", "temp": "numeric",
                "sps": "numeric", "aps": "numeric",
                "scoma": "numeric", "totcst": "numeric", "totmcst": "numeric",
                "avtisst": "numeric", "surv2m": "numeric", "surv6m": "numeric",
                "hday": "numeric", "prg2m": "numeric", "prg6m": "numeric",
                "wblc": "numeric", "pafi": "numeric", "alb": "numeric",
                "bili": "numeric", "crea": "numeric", "sod": "numeric",
                "ph": "numeric", "glucose": "numeric", "bun": "numeric",
                "urine": "numeric", "adlp": "numeric", "adls": "numeric",
                "adlsc": "numeric", "edu": "numeric", "num.co": "numeric",
                "dnrday": "numeric",
            },
            "redundant_features": [], "leakage_cols": [],
        },
    },
}

_MODEL_TO_DOWNSTREAM = {
    "LogReg": "LogisticRegression", "Ridge": "Ridge",
    "KNN": "KNN", "RF": "RandomForest", "GBM": "GradientBoosting",
}


class C4SafetyStep(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy()
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        self._obj_cols = obj_cols
        if obj_cols:
            df[obj_cols] = df[obj_cols].fillna("__missing__").astype(str)
            self._enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
            self._enc.fit(df[obj_cols])
        else:
            self._enc = None
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        num_nan_cols = [c for c in df.columns if df[c].isna().any()]
        self._num_nan_cols = num_nan_cols
        if num_nan_cols:
            self._imp = SimpleImputer(strategy="median")
            self._imp.fit(df[num_nan_cols])
        else:
            self._imp = None
        self._fit_columns = list(df.columns)
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(X).copy()
        if self._enc is not None:
            cols_present = [c for c in self._obj_cols if c in df.columns]
            if cols_present:
                df[cols_present] = self._enc.transform(df[cols_present].fillna("__missing__").astype(str))
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if self._imp is not None:
            cols_present = [c for c in self._num_nan_cols if c in df.columns]
            if cols_present:
                df[cols_present] = self._imp.transform(df[cols_present])
        df = df.fillna(0)
        return df.values


def get_models(task):
    if task == "classification":
        return {
            "LogReg": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "RF":     RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
            "KNN":    KNeighborsClassifier(n_neighbors=5),
            "GBM":    GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        }
    else:
        return {
            "Ridge": Ridge(random_state=RANDOM_STATE),
            "RF":    RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
            "KNN":   KNeighborsRegressor(n_neighbors=5),
            "GBM":   GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
        }


def run_c4_llm(llm_tag, dataset_filter=None, model_filter=None):
    results = []
    for dataset, spec in PHASE_C.items():
        if dataset_filter and dataset != dataset_filter:
            continue
        prov_dir = EXPORTS / dataset / "provenance"
        plan_path = prov_dir / f"c4_plan_{llm_tag}.json"
        if not plan_path.exists():
            print(f"  [SKIP] {dataset}: no plan at {plan_path}")
            continue
        df = pd.read_csv(EXPORTS / dataset / "c0_raw.csv")
        plan_raw = json.loads(plan_path.read_text())
        target = spec["target"]
        task = spec["task"]
        cardinality = compute_cardinality(df)
        X = df.drop(columns=[target], errors="ignore")
        y = df[target].copy()
        if task == "classification":
            cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            scoring = {"f1_macro": make_scorer(f1_score, average="macro", zero_division=0)}
            main_metric = "test_f1_macro"; metric_name = "f1_macro"
        else:
            cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            scoring = {"r2": make_scorer(r2_score)}
            main_metric = "test_r2"; metric_name = "r2"
        print(f"  [{dataset}]")
        for model_name, model in get_models(task).items():
            if model_filter and model_name != model_filter:
                continue
            t0 = time.time()
            try:
                downstream = _MODEL_TO_DOWNSTREAM.get(model_name, "RandomForest")
                ctx = {**spec["user_context"], "downstream_model": downstream}
                plan_enforced, changes = enforce_plan(copy.deepcopy(plan_raw), ctx, cardinality)
                pipe = Pipeline([
                    ("cleaner", PlanBasedCleaner(plan=plan_enforced, target_column=target)),
                    ("safety", C4SafetyStep()),
                    ("model", model),
                ])
                scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, error_score="raise")
                vals = scores[main_metric]
                row = {"dataset": dataset, "condition": f"C4_{llm_tag}", "model": model_name,
                       "task_type": task, "n_folds": N_FOLDS, "mean": float(np.mean(vals)),
                       "std": float(np.std(vals)), "metric": metric_name}
                print(f"    {model_name}: {row['mean']:.6f} +/-{row['std']:.6f} ({time.time()-t0:.1f}s)")
                results.append(row)
            except Exception as e:
                print(f"    {model_name}: ERROR - {e}")
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluation/run_c4_one_llm.py <llm_tag> [dataset] [model]")
        sys.exit(1)
    llm_tag = sys.argv[1]
    dataset_filter = sys.argv[2] if len(sys.argv) > 2 else None
    model_filter = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"=== C4: {llm_tag} dataset={dataset_filter} model={model_filter} ===")
    results = run_c4_llm(llm_tag, dataset_filter=dataset_filter, model_filter=model_filter)
    df = pd.DataFrame(results)

    # Build output filename
    suffix_parts = [f"c4_{llm_tag}"]
    if dataset_filter:
        suffix_parts.append(dataset_filter)
    if model_filter:
        suffix_parts.append(model_filter)
    suffix = "_".join(suffix_parts)
    out_file = OUT / f"phase_c_partial_{suffix}.csv"
    df.to_csv(out_file, index=False)
    print(f"Saved {len(df)} rows to {out_file.name}")
