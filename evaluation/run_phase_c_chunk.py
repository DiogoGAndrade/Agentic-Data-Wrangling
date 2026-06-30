"""
Chunked Phase C evaluation. Run with --chunk argument.
Chunks: c0_platform, c0_support2_clf, c0_support2_reg,
        c4_qwen2_5_3b, c4_llama3_2_3b, c4_mistral_7b, c4_qwen2_5_7b,
        c4_llama3_1_8b, c4_gemma2_9b, c4_mistral-nemo_12b, c4_qwen2_5_14b
        merge (final merge of all partial CSVs into PHASE_C_RESULTS_CURATED.csv)
"""
import argparse, copy, json, sys, warnings, time
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
        if col == target:
            continue
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


def run_c0(dataset):
    spec = PHASE_C[dataset]
    target = spec["target"]
    task = spec["task"]
    df = pd.read_csv(EXPORTS / dataset / "c0_raw.csv")
    X = df.drop(columns=[target])
    y = df[target].copy()
    X_enc = encode_categoricals(X, target)
    X_enc = X_enc.fillna(X_enc.median(numeric_only=True))
    for col in X_enc.select_dtypes(include="object").columns:
        X_enc[col] = X_enc[col].astype("category").cat.codes
    if task == "classification":
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scoring = {"f1_macro": make_scorer(f1_score, average="macro", zero_division=0)}
        main_metric = "test_f1_macro"; metric_name = "f1_macro"
    else:
        cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scoring = {"r2": make_scorer(r2_score)}
        main_metric = "test_r2"; metric_name = "r2"
    results = []
    for model_name, model in get_models(task).items():
        t0 = time.time()
        try:
            pipe = Pipeline([("model", model)])
            scores = cross_validate(pipe, X_enc, y, cv=cv, scoring=scoring, error_score="raise")
            vals = scores[main_metric]
            row = {"dataset": dataset, "condition": "C0", "model": model_name, "task_type": task,
                   "n_folds": N_FOLDS, "mean": float(np.mean(vals)), "std": float(np.std(vals)), "metric": metric_name}
            print(f"  {dataset}/{model_name}: {row['mean']:.6f} +/-{row['std']:.6f} ({time.time()-t0:.1f}s)")
            results.append(row)
        except Exception as e:
            print(f"  {dataset}/{model_name}: ERROR - {e}")
    return results


def run_c4_llm(llm_tag):
    results = []
    for dataset, spec in PHASE_C.items():
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


def merge_all():
    partial_files = sorted(OUT.glob("phase_c_partial_*.csv"))
    all_dfs = []
    for f in partial_files:
        df = pd.read_csv(f)
        all_dfs.append(df)
        print(f"  {f.name}: {len(df)} rows")
    if not all_dfs:
        print("No partial files found!")
        return
    df_out = pd.concat(all_dfs, ignore_index=True)
    # deduplicate keeping last
    df_out = df_out.drop_duplicates(subset=["dataset", "condition", "model"], keep="last")
    out_path = OUT / "PHASE_C_RESULTS_CURATED.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved {len(df_out)} rows to {out_path}")
    return df_out


def compute_wtl(df_out):
    c0 = df_out[df_out["condition"] == "C0"].set_index(["dataset", "model"])
    c4 = df_out[df_out["condition"] != "C0"]
    wins, ties, losses = 0, 0, 0
    for _, row in c4.iterrows():
        key = (row["dataset"], row["model"])
        if key not in c0.index:
            continue
        baseline = c0.loc[key]
        delta = row["mean"] - baseline["mean"]
        sigma = baseline["std"]
        if delta > sigma:
            wins += 1
        elif delta < -sigma:
            losses += 1
        else:
            ties += 1
    print(f"W/T/L = {wins}/{ties}/{losses}")
    return wins, ties, losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", required=True,
                        help="c0_platform | c0_support2_clf | c0_support2_reg | c4_<llm_tag> | merge | wtl")
    parser.add_argument("--model", default=None,
                        help="Run only this model (LogReg|Ridge|RF|KNN|GBM). For incremental runs.")
    args = parser.parse_args()
    chunk = args.chunk
    model_filter = args.model

    if chunk.startswith("c0_"):
        dataset = chunk[3:]
        print(f"=== C0: {dataset} ===")
        results = run_c0(dataset)
        df = pd.DataFrame(results)
        if model_filter:
            df = df[df["model"] == model_filter]
            suffix = f"phase_c_partial_{chunk}_{model_filter}.csv"
        else:
            suffix = f"phase_c_partial_{chunk}.csv"
        df.to_csv(OUT / suffix, index=False)
        print(f"Saved {len(df)} rows to {suffix}")

    elif chunk.startswith("c4_"):
        llm_tag = chunk[3:]
        print(f"=== C4: {llm_tag} ===")
        results = run_c4_llm(llm_tag)
        df = pd.DataFrame(results)
        if model_filter:
            df = df[df["model"] == model_filter]
            suffix = f"phase_c_partial_{chunk}_{model_filter}.csv"
        else:
            suffix = f"phase_c_partial_{chunk}.csv"
        df.to_csv(OUT / suffix, index=False)
        print(f"Saved {len(df)} rows to {suffix}")

    elif chunk == "merge":
        df_out = merge_all()
        if df_out is not None:
            compute_wtl(df_out)

    elif chunk == "wtl":
        out_path = OUT / "PHASE_C_RESULTS_CURATED.csv"
        if not out_path.exists():
            print("PHASE_C_RESULTS_CURATED.csv not found, run merge first")
        else:
            df_out = pd.read_csv(out_path)
            compute_wtl(df_out)
