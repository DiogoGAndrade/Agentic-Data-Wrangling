"""
Ablation: guardrails applied to an EMPTY plan (no LLM) on the challenging tier.

Question answered: "If enforcement converges all 8 LLM plans, is the LLM needed at all?"
This runs the identical C4 evaluation path (enforce_plan + PlanBasedCleaner +
C4SafetyStep + optional G12 scaler) but with plan_raw = {"actions": []}.

Outputs evaluation/outputs/ABLATION_EMPTY_PLAN.csv and prints a W/T/L
comparison (|delta| > sigma_C0) of:
  - empty-plan-guardrails vs C0
  - empty-plan-guardrails vs C4 (from PHASE_C_RESULTS_CURATED.csv)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse the exact evaluation machinery from the curated Phase C run.
# run_phase_c_curated executes at import time, so we import its module
# pieces manually instead.
import copy
import warnings
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
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

EXPORTS = ROOT / "data" / "exports"
OUT = ROOT / "evaluation" / "outputs"
N_FOLDS = 5

PHASE_C = {
    "platform": {"target": "purchased", "task": "classification"},
    "support2_clf": {"target": "hospdead", "task": "classification"},
    "support2_reg": {"target": "log_charges", "task": "regression"},
}

_MODEL_TO_DOWNSTREAM = {
    "LogReg": "LogisticRegression", "Ridge": "Ridge",
    "KNN": "KNN", "RF": "RandomForest", "GBM": "GradientBoosting",
}


class C4SafetyStep(BaseEstimator, TransformerMixin):
    """Identical to run_phase_c_curated.C4SafetyStep."""

    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy()
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        self._obj_cols = obj_cols
        if obj_cols:
            df[obj_cols] = df[obj_cols].fillna("__missing__").astype(str)
            self._enc = OrdinalEncoder(handle_unknown="use_encoded_value",
                                       unknown_value=-1, encoded_missing_value=-1)
            self._enc.fit(df[obj_cols])
            # NOTE: fix of a latent bug in the original C4SafetyStep — encode the
            # object columns here too, so the NaN scan below sees the encoded
            # values (the original never hit this path because the C4 plans
            # always encoded categoricals before the safety step).
            df[obj_cols] = self._enc.transform(df[obj_cols])
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
                df[cols_present] = self._enc.transform(
                    df[cols_present].fillna("__missing__").astype(str))
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
            "RF": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "GBM": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        }
    return {
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "RF": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "GBM": GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-summary", action="store_true")
    args = ap.parse_args()
    results = []
    for dataset, spec in PHASE_C.items():
        if args.dataset and dataset != args.dataset:
            continue
        print(f"=== {dataset} (empty plan + guardrails) ===")
        df = pd.read_csv(EXPORTS / dataset / "c0_raw.csv")
        target, task = spec["target"], spec["task"]
        cardinality = compute_cardinality(df)
        X = df.drop(columns=[target], errors="ignore")
        y = df[target].copy()
        if task == "classification":
            cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            scoring = {"m": make_scorer(f1_score, average="macro", zero_division=0)}
            metric_name = "f1_macro"
        else:
            cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            scoring = {"m": make_scorer(r2_score)}
            metric_name = "r2"
        for model_name, model in get_models(task).items():
            if args.model and model_name != args.model:
                continue
            downstream = _MODEL_TO_DOWNSTREAM.get(model_name, "RandomForest")
            if task == "regression" and model_name == "KNN":
                downstream += "_regression"
            ctx = {"downstream_model": downstream,
                   "redundant_features": [], "leakage_cols": []}
            plan_raw = {"actions": []}  # <- the whole point: NO LLM plan
            plan_enforced, changes = enforce_plan(copy.deepcopy(plan_raw), ctx, cardinality)
            has_g12 = any(a.get("action") == "scale_features"
                          for a in plan_enforced.get("actions", []))
            if has_g12:
                plan_enforced["actions"] = [a for a in plan_enforced["actions"]
                                            if a.get("action") != "scale_features"]
            steps = [("cleaner", PlanBasedCleaner(plan=plan_enforced, target_column=target)),
                     ("safety", C4SafetyStep())]
            if has_g12:
                steps.append(("scaler", StandardScaler()))
            steps.append(("model", model))
            pipe = Pipeline(steps)
            try:
                scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, error_score="raise", n_jobs=-1)
                vals = scores["test_m"]
                row = {"dataset": dataset, "condition": "GUARDRAILS_ONLY",
                       "model": model_name, "task_type": task, "n_folds": N_FOLDS,
                       "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                       "metric": metric_name,
                       "enforcement_changes": ";".join(changes) if changes else ""}
                print(f"   {model_name}: {row['mean']:.4f} +-{row['std']:.4f} "
                      f"| injected: {row['enforcement_changes'] or 'nothing'}")
                results.append(row)
            except Exception as e:
                print(f"   {model_name}: ERROR -- {e}")

    out = pd.DataFrame(results)
    csv_path = OUT / "ABLATION_EMPTY_PLAN.csv"
    if csv_path.exists() and csv_path.stat().st_size > 10:
        prev = pd.read_csv(csv_path)
        out = pd.concat([prev, out], ignore_index=True).drop_duplicates(
            subset=["dataset", "model"], keep="last")
    out.to_csv(csv_path, index=False)
    if args.no_summary:
        return

    # W/T/L vs C0 and vs C4 from the curated results
    cur = pd.read_csv(OUT / "PHASE_C_RESULTS_CURATED.csv")
    c0 = cur[cur.condition == "C0"].set_index(["dataset", "model"])
    c4 = (cur[cur.condition.str.startswith("C4")]
          .groupby(["dataset", "model"])["mean"].mean())  # identical across LLMs
    print("\n=== Guardrails-only vs C0 and vs C4 (threshold |d| > sigma_C0) ===")
    print(f"{'cell':32s} {'C0':>8s} {'GRDonly':>8s} {'C4':>8s} {'vsC0':>6s} {'vsC4':>6s}")
    out = pd.read_csv(csv_path)
    for _, r in out.iterrows():
        key = (r.dataset, r.model)
        if key not in c0.index:
            continue
        base = c0.loc[key]
        thr = base["std"]
        d0 = r["mean"] - base["mean"]
        v0 = "WIN" if d0 > thr else ("LOSS" if d0 < -thr else "TIE")
        d4 = r["mean"] - c4.loc[key]
        v4 = "BELOW" if d4 < -thr else ("ABOVE" if d4 > thr else "EQUAL")
        print(f"{r.dataset+'/'+r.model:32s} {base['mean']:8.4f} {r['mean']:8.4f} "
              f"{c4.loc[key]:8.4f} {v0:>6s} {v4:>6s}")


if __name__ == "__main__":
    main()
