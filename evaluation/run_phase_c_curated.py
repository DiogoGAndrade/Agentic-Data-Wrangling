"""
Phase C re-evaluation using curated plans.
Does not call Ollama; uses the plan JSON files already on disk (curated and original).
"""
import copy, json, sys, warnings
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
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
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

LLM_ALL = [
    ("qwen2.5:3b", "qwen2_5_3b"),
    ("llama3.2:3b", "llama3_2_3b"),
    ("mistral:7b", "mistral_7b"),
    ("qwen2.5:7b", "qwen2_5_7b"),
    ("llama3.1:8b", "llama3_1_8b"),
    ("gemma2:9b", "gemma2_9b"),
    ("mistral-nemo:12b", "mistral-nemo_12b"),
    ("qwen2.5:14b", "qwen2_5_14b"),
]

_MODEL_TO_DOWNSTREAM = {
    "LogReg": "LogisticRegression", "Ridge": "Ridge",
    "KNN": "KNN", "RF": "RandomForest", "GBM": "GradientBoosting",
}

def encode_categoricals(df, target):
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        if col == target: continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

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
            "RF":     RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
            "KNN":    KNeighborsClassifier(n_neighbors=5),
            "GBM":    GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        }
    else:
        return {
            "Ridge": Ridge(random_state=RANDOM_STATE),
            "RF":    RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
            "KNN":   KNeighborsRegressor(n_neighbors=5),
            "GBM":   GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
        }

def evaluate_c4(df, plan_raw, dataset, llm_tag, user_context_base):
    spec = PHASE_C[dataset]
    target = spec["target"]
    task = spec["task"]
    cardinality = compute_cardinality(df)
    X = df.drop(columns=[target], errors="ignore")
    y = df[target].copy()
    if task == "classification":
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scoring = {"f1_macro": make_scorer(f1_score, average="macro", zero_division=0)}
        main_metric = "test_f1_macro"
        metric_name = "f1_macro"
    else:
        cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scoring = {"r2": make_scorer(r2_score)}
        main_metric = "test_r2"
        metric_name = "r2"
    results = []
    for model_name, model in get_models(task).items():
        try:
            downstream = _MODEL_TO_DOWNSTREAM.get(model_name, "RandomForest")
            # G11 distinguishes KNN clf from KNN regressor via "regress" in downstream string
            if task == "regression" and model_name == "KNN":
                downstream = downstream + "_regression"
            ctx = {**user_context_base, "downstream_model": downstream}
            plan_enforced, changes = enforce_plan(copy.deepcopy(plan_raw), ctx, cardinality)
            # G12: scale_features is injected into the plan for linear models, but
            # C4SafetyStep must run after the cleaner (it re-encodes residual object cols
            # and fills NaN). Scaling INSIDE the cleaner gets undone by SafetyStep.
            # Solution: if G12 was injected, strip it from the plan and add a StandardScaler
            # AFTER SafetyStep in the sklearn Pipeline — same effect, correct order.
            from sklearn.preprocessing import StandardScaler as _SS
            has_g12 = any(a.get("action") == "scale_features" for a in plan_enforced.get("actions", []))
            if has_g12:
                plan_enforced["actions"] = [
                    a for a in plan_enforced["actions"] if a.get("action") != "scale_features"
                ]
            pipe_steps = [
                ("cleaner", PlanBasedCleaner(plan=plan_enforced, target_column=target)),
                ("safety", C4SafetyStep()),
            ]
            if has_g12:
                pipe_steps.append(("scaler", _SS()))
            pipe_steps.append(("model", model))
            pipe = Pipeline(pipe_steps)
            scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, error_score="raise")
            vals = scores[main_metric]
            row = {
                "dataset": dataset, "condition": f"C4_{llm_tag}",
                "model": model_name, "task_type": task,
                "n_folds": N_FOLDS, "mean": float(np.mean(vals)),
                "std": float(np.std(vals)), "metric": metric_name,
            }
            print(f"      {model_name}: {row['mean']:.4f} +-{row['std']:.4f}")
            results.append(row)
        except Exception as e:
            print(f"      {model_name}: ERROR -- {e}")
    return results

all_results = []

# C0 baselines (unchanged)
print("=== C0 Baselines ===")
for dataset, spec in PHASE_C.items():
    df = pd.read_csv(EXPORTS / dataset / "c0_raw.csv")
    target = spec["target"]
    task = spec["task"]
    X = df.drop(columns=[target])
    y = df[target].copy()
    X_enc = encode_categoricals(X, target)
    X_enc = X_enc.fillna(X_enc.median(numeric_only=True))
    for col in X_enc.select_dtypes(include="object").columns:
        X_enc[col] = X_enc[col].astype("category").cat.codes
    if task == "classification":
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scoring = {"f1_macro": make_scorer(f1_score, average="macro", zero_division=0)}
        main_metric = "test_f1_macro"
        metric_name = "f1_macro"
    else:
        cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scoring = {"r2": make_scorer(r2_score)}
        main_metric = "test_r2"
        metric_name = "r2"
    for model_name, model in get_models(task).items():
        try:
            pipe = Pipeline([("model", model)])
            scores = cross_validate(pipe, X_enc, y, cv=cv, scoring=scoring, error_score="raise")
            vals = scores[main_metric]
            row = {
                "dataset": dataset, "condition": "C0",
                "model": model_name, "task_type": task,
                "n_folds": N_FOLDS, "mean": float(np.mean(vals)),
                "std": float(np.std(vals)), "metric": metric_name,
            }
            print(f"  {dataset}/{model_name}: {row['mean']:.4f}")
            all_results.append(row)
        except Exception as e:
            print(f"  {dataset}/{model_name}: ERROR -- {e}")

# C4 per LLM
print("\n=== C4 Evaluation ===")
for llm_full, llm_tag in LLM_ALL:
    print(f"\n[LLM] {llm_full}")
    for dataset, spec in PHASE_C.items():
        prov_dir = EXPORTS / dataset / "provenance"
        plan_path = prov_dir / f"c4_plan_{llm_tag}.json"
        if not plan_path.exists():
            print(f"  [SKIP] {dataset}: no plan")
            continue
        df = pd.read_csv(EXPORTS / dataset / "c0_raw.csv")
        plan_raw = json.loads(plan_path.read_text())
        print(f"  [{dataset}]")
        rows = evaluate_c4(df, plan_raw, dataset, llm_tag, spec["user_context"])
        all_results.extend(rows)

# Save
df_out = pd.DataFrame(all_results)
out_path = OUT / "PHASE_C_RESULTS_CURATED.csv"
df_out.to_csv(out_path, index=False)
print(f"\n[SAVED] {out_path}")
print(f"Total rows: {len(df_out)}")
