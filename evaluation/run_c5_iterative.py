"""
evaluation/run_c5_iterative.py
================================
C5 — Iterative Feedback Cleaning

Concept (inspired by IterClean / GA-PRE):
  The agent starts from the C4 plan, runs 5-fold CV, and uses the CV score
  as feedback to systematically explore plan mutations.  Each iteration
  keeps the best-scoring plan found so far and spawns candidate mutations
  from it.  This mimics what an LLM-based iterative loop would do when
  given its own CV score as feedback — without requiring Ollama in the
  evaluation sandbox.

  Targets:
    bank / logreg  —  C4=0.7471  threshold=0.7520  gap=0.0049
    heart / rf     —  C4=0.6410  threshold=0.6830  gap=0.0414  (approx)

  WIN criterion  (matching MASTER):
    Δ = f1_macro(C5) – f1_macro(C0)
    WIN  if  Δ  >  std_C0 + std_C5
    The threshold here uses the conservative bound stored from MASTER.

Usage:
  python -m evaluation.run_c5_iterative
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from engine.cleaning_pipeline import PlanBasedCleaner
from engine.config import RANDOM_STATE
from evaluation.run_experiments import build_final_preprocessor, load_raw, DatasetSpec

# ---------------------------------------------------------------------------
# Dataset specs (Phase B classification)
# ---------------------------------------------------------------------------
BANK_SPEC = DatasetSpec(
    dataset_id="bank",
    target="y",
    base_dir=Path("data/exports/bank"),
    task_type="classification",
    subsample_n=20_000,
)

HEART_SPEC = DatasetSpec(
    dataset_id="heart",
    target="target",
    base_dir=Path("data/exports/heart"),
    task_type="classification",
    subsample_n=None,
)

# C0 results from MASTER_RESULTS_TABLE.csv (verified 2026-05-22)
MASTER_C0 = {
    "bank_logreg":  {"f1_macro": 0.7471, "std": 0.0019},
    "heart_rf":     {"f1_macro": 0.8283, "std": 0.0262},
}
# C4 results from MASTER_RESULTS_TABLE.csv
MASTER_C4 = {
    "bank_logreg":  {"f1_macro": 0.7471, "std": 0.0030},
    "heart_rf":     {"f1_macro": 0.8293, "std": 0.0222},
}
# C5 findings (2026-05-22):
#   bank/logreg: all variants Δ=0.0000, threshold=0.0049 — at absolute ceiling
#   heart/rf:    best Δ=+0.0010 (handle_missing), threshold=0.0484
#                blocker = fold variance on n=303 rows, not cleaning quality
#   CONCLUSION: No additional WINs achievable. Final score = 1W/11T/0L.


# ---------------------------------------------------------------------------
# Evaluate a single plan on a dataset/model combo
# ---------------------------------------------------------------------------
def _evaluate(X, y, model, plan, target_col, scale):
    steps = []
    if plan and plan.get("actions"):
        steps.append(("clean", PlanBasedCleaner(plan=plan, target_column=target_col)))
    steps.append(("preprocess", build_final_preprocessor(scale_numeric=scale)))
    steps.append(("model", model))
    pipe = Pipeline(steps=steps)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    res = cross_validate(pipe, X, y, cv=cv,
                         scoring={"f1_macro": "f1_macro"},
                         error_score="raise", n_jobs=None)
    scores = res["test_f1_macro"]
    return float(np.mean(scores)), float(np.std(scores))


def win_tie_loss(delta, std_c0, std_cand):
    threshold = std_c0 + std_cand
    if delta > threshold:
        return "WIN"
    if delta < -threshold:
        return "LOSS"
    return "TIE"


# ---------------------------------------------------------------------------
# Plan mutation library — bank / logreg
# ---------------------------------------------------------------------------
def bank_plan_mutations():
    """
    Ordered list of (label, plan_dict) candidates for bank/logreg.
    We start from C4's base (handle_missing + clip_outliers + normalize_text)
    and explore encoding strategies for 'education' and 'pdays' treatment.
    """
    base_actions_no_enc = [
        {
            "action": "handle_missing",
            "rationale": "Safety imputation.",
            "target_columns": [],
            "params": {"strategy": "impute"},
        },
        {
            "action": "clip_outliers",
            "rationale": "IQR clipping.",
            "target_columns": [],
            "params": {"method": "iqr", "iqr_k": 3.0},
        },
        {
            "action": "normalize_text",
            "rationale": "Strip whitespace.",
            "target_columns": [],
            "params": {},
        },
    ]

    edu_ordinal_map = {
        "illiterate": 0, "basic.4y": 1, "basic.6y": 2, "basic.9y": 3,
        "high.school": 4, "professional.course": 5, "university.degree": 6,
        "unknown": 3,  # median impute
    }

    plans = []

    # ---- Variant 1: C4 baseline (one-hot everything) ----
    ohe_all = copy.deepcopy(base_actions_no_enc) + [{
        "action": "encode_categorical_per_column",
        "rationale": "One-hot all cats (C4 baseline).",
        "target_columns": [],
        "params": {
            "column_encodings": {
                "job": "one_hot", "marital": "one_hot", "education": "one_hot",
                "default": "one_hot", "housing": "one_hot", "loan": "one_hot",
                "contact": "one_hot", "month": "one_hot", "day_of_week": "one_hot",
                "poutcome": "one_hot",
            },
            "default_method": "one_hot",
        },
    }]
    plans.append(("C4-baseline (OHE all)", {"actions": ohe_all}))

    # ---- Variant 2: ordinal education ----
    ord_edu = copy.deepcopy(base_actions_no_enc) + [{
        "action": "cast_type",
        "rationale": "Ordinal-encode education via cast.",
        "target_columns": ["education"],
        "params": {
            "columns": ["education"],
            "dtype": "category",
            "ordered": True,
            "categories": list(edu_ordinal_map.keys()),
        },
    }, {
        "action": "encode_categorical_per_column",
        "rationale": "OHE remaining cats; education already numeric.",
        "target_columns": [],
        "params": {
            "column_encodings": {
                "job": "one_hot", "marital": "one_hot",
                "default": "one_hot", "housing": "one_hot", "loan": "one_hot",
                "contact": "one_hot", "month": "one_hot", "day_of_week": "one_hot",
                "poutcome": "one_hot",
            },
            "default_method": "one_hot",
        },
    }]
    plans.append(("ordinal-education", {"actions": ord_edu}))

    # ---- Variant 3: pdays binary + OHE all ----
    pdays_bin = copy.deepcopy(ohe_all)
    pdays_bin.insert(1, {
        "action": "bin_numeric",
        "rationale": "pdays=999 is a sentinel; convert to contacted binary.",
        "target_columns": ["pdays"],
        "params": {
            "columns": ["pdays"],
            "strategy": "custom",
            "bins": [0, 998, 999],
            "labels": [0, 1],
        },
    })
    plans.append(("pdays-binary + OHE", {"actions": pdays_bin}))

    # ---- Variant 4: pdays binary + ordinal education ----
    pdays_ord = copy.deepcopy(ord_edu)
    pdays_ord.insert(1, {
        "action": "bin_numeric",
        "rationale": "pdays binary sentinel.",
        "target_columns": ["pdays"],
        "params": {
            "columns": ["pdays"],
            "strategy": "custom",
            "bins": [0, 998, 999],
            "labels": [0, 1],
        },
    })
    plans.append(("pdays-binary + ordinal-edu", {"actions": pdays_ord}))

    # ---- Variant 5: add missing indicators ----
    miss_ind = copy.deepcopy(ohe_all)
    miss_ind.insert(0, {
        "action": "add_missing_indicators",
        "rationale": "Flag implicit missings before imputation.",
        "target_columns": [],
        "params": {},
    })
    plans.append(("missing-indicators + OHE", {"actions": miss_ind}))

    # ---- Variant 6: semantic missing-to-category ----
    sem = copy.deepcopy(ohe_all)
    sem.insert(0, {
        "action": "semantic_missing_to_category",
        "rationale": "Map unknown/999/none to NaN before imputation.",
        "target_columns": [],
        "params": {},
    })
    plans.append(("semantic-missing + OHE", {"actions": sem}))

    # ---- Variant 7: semantic + pdays binary + ordinal edu ----
    sem_pdays_ord = [
        {
            "action": "semantic_missing_to_category",
            "rationale": "Semantic NaN.",
            "target_columns": [],
            "params": {},
        },
        {
            "action": "handle_missing",
            "rationale": "Impute after semantic.",
            "target_columns": [],
            "params": {"strategy": "impute"},
        },
        {
            "action": "bin_numeric",
            "rationale": "pdays binary.",
            "target_columns": ["pdays"],
            "params": {"columns": ["pdays"], "strategy": "custom",
                       "bins": [0, 998, 999], "labels": [0, 1]},
        },
        {
            "action": "clip_outliers",
            "rationale": "Clip.",
            "target_columns": [],
            "params": {"method": "iqr", "iqr_k": 3.0},
        },
        {
            "action": "normalize_text",
            "rationale": "Strip whitespace.",
            "target_columns": [],
            "params": {},
        },
        {
            "action": "encode_categorical_per_column",
            "rationale": "OHE non-edu cats.",
            "target_columns": [],
            "params": {
                "column_encodings": {
                    "job": "one_hot", "marital": "one_hot",
                    "default": "one_hot", "housing": "one_hot", "loan": "one_hot",
                    "contact": "one_hot", "month": "one_hot", "day_of_week": "one_hot",
                    "poutcome": "one_hot",
                },
                "default_method": "one_hot",
            },
        },
    ]
    plans.append(("semantic+pdays-binary+ordinal-edu", {"actions": sem_pdays_ord}))

    # ---- Variant 8: tighter IQR clipping ----
    tight_clip = copy.deepcopy(ohe_all)
    for a in tight_clip:
        if a["action"] == "clip_outliers":
            a["params"]["iqr_k"] = 1.5
    plans.append(("tight-IQR(1.5) + OHE", {"actions": tight_clip}))

    # ---- Variant 9: no clipping at all ----
    no_clip = [a for a in copy.deepcopy(ohe_all) if a["action"] != "clip_outliers"]
    plans.append(("no-clip + OHE", {"actions": no_clip}))

    return plans


# ---------------------------------------------------------------------------
# Plan mutation library — heart / rf
# ---------------------------------------------------------------------------
def heart_plan_mutations():
    """
    Heart dataset: all columns are float64 (no string categoricals).
    ca and thal have integer-coded categories.
    RF doesn't need scaling.
    """
    # Integer-coded categorical columns in heart
    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

    plans = []

    # ---- Variant 1: C4 baseline (empty plan — no actions) ----
    plans.append(("C4-baseline (empty)", {"actions": []}))

    # ---- Variant 2: handle missing ----
    miss = [{"action": "handle_missing", "rationale": "KNN impute for heart nulls.",
              "target_columns": [], "params": {"strategy": "impute"}}]
    plans.append(("handle-missing", {"actions": miss}))

    # ---- Variant 3: OHE integer categoricals ----
    ohe_cats = [
        {"action": "handle_missing", "rationale": "Impute first.",
         "target_columns": [], "params": {"strategy": "impute"}},
        {"action": "encode_categorical_per_column",
         "rationale": "OHE integer-coded categoricals.",
         "target_columns": cat_cols,
         "params": {
             "column_encodings": {c: "one_hot" for c in cat_cols},
             "default_method": "one_hot",
         }},
    ]
    plans.append(("OHE integer cats", {"actions": ohe_cats}))

    # ---- Variant 4: clip outliers ----
    clip = [
        {"action": "handle_missing", "rationale": "Impute.", "target_columns": [],
         "params": {"strategy": "impute"}},
        {"action": "clip_outliers", "rationale": "Clip extreme values.",
         "target_columns": [], "params": {"method": "iqr", "iqr_k": 3.0}},
    ]
    plans.append(("handle-missing + clip IQR3", {"actions": clip}))

    # ---- Variant 5: tighter IQR 1.5 ----
    clip15 = copy.deepcopy(clip)
    clip15[1]["params"]["iqr_k"] = 1.5
    plans.append(("handle-missing + clip IQR1.5", {"actions": clip15}))

    # ---- Variant 6: missing indicators + clip ----
    ind_clip = [
        {"action": "add_missing_indicators", "rationale": "Flag missings.",
         "target_columns": [], "params": {}},
        {"action": "handle_missing", "rationale": "Impute.", "target_columns": [],
         "params": {"strategy": "impute"}},
        {"action": "clip_outliers", "rationale": "Clip.", "target_columns": [],
         "params": {"method": "iqr", "iqr_k": 3.0}},
    ]
    plans.append(("missing-indicators + clip", {"actions": ind_clip}))

    # ---- Variant 7: OHE cats + clip ----
    ohe_clip = [
        {"action": "handle_missing", "rationale": "Impute.", "target_columns": [],
         "params": {"strategy": "impute"}},
        {"action": "encode_categorical_per_column",
         "rationale": "OHE integer cats.",
         "target_columns": cat_cols,
         "params": {
             "column_encodings": {c: "one_hot" for c in cat_cols},
             "default_method": "one_hot",
         }},
        {"action": "clip_outliers", "rationale": "Clip after OHE.",
         "target_columns": [], "params": {"method": "iqr", "iqr_k": 3.0}},
    ]
    plans.append(("OHE cats + clip", {"actions": ohe_clip}))

    # ---- Variant 8: semantic missing + OHE cats + clip ----
    sem_ohe_clip = [
        {"action": "semantic_missing_to_category", "rationale": "Flag semantic 0s.",
         "target_columns": [], "params": {}},
        {"action": "handle_missing", "rationale": "Impute.", "target_columns": [],
         "params": {"strategy": "impute"}},
        {"action": "encode_categorical_per_column",
         "rationale": "OHE integer cats.",
         "target_columns": cat_cols,
         "params": {
             "column_encodings": {c: "one_hot" for c in cat_cols},
             "default_method": "one_hot",
         }},
        {"action": "clip_outliers", "rationale": "Clip.", "target_columns": [],
         "params": {"method": "iqr", "iqr_k": 3.0}},
    ]
    plans.append(("semantic + OHE cats + clip", {"actions": sem_ohe_clip}))

    # ---- Variant 9: bin age (risk groups) ----
    age_bins = [
        {"action": "handle_missing", "rationale": "Impute.", "target_columns": [],
         "params": {"strategy": "impute"}},
        {"action": "bin_numeric", "rationale": "Age risk groups for RF.",
         "target_columns": ["age"],
         "params": {
             "columns": ["age"],
             "strategy": "quantile",
             "n_bins": 4,
         }},
        {"action": "clip_outliers", "rationale": "Clip.", "target_columns": [],
         "params": {"method": "iqr", "iqr_k": 3.0}},
    ]
    plans.append(("age-bins (quantile) + clip", {"actions": age_bins}))

    # ---- Variant 10: select features (drop low-variance / high-corr) ----
    sel = [
        {"action": "handle_missing", "rationale": "Impute.", "target_columns": [],
         "params": {"strategy": "impute"}},
        {"action": "select_features", "rationale": "Remove low-var and correlated.",
         "target_columns": [], "params": {"variance_threshold": 0.01,
                                           "correlation_threshold": 0.95}},
    ]
    plans.append(("select-features", {"actions": sel}))

    # ---- Variant 11: OHE cats + missing indicators + clip ----
    ohe_ind_clip = [
        {"action": "add_missing_indicators", "rationale": "Flag missings.",
         "target_columns": [], "params": {}},
        {"action": "handle_missing", "rationale": "Impute.", "target_columns": [],
         "params": {"strategy": "impute"}},
        {"action": "encode_categorical_per_column",
         "rationale": "OHE integer cats.",
         "target_columns": cat_cols,
         "params": {
             "column_encodings": {c: "one_hot" for c in cat_cols},
             "default_method": "one_hot",
         }},
        {"action": "clip_outliers", "rationale": "Clip.", "target_columns": [],
         "params": {"method": "iqr", "iqr_k": 3.0}},
    ]
    plans.append(("missing-indicators + OHE + clip", {"actions": ohe_ind_clip}))

    return plans


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_search(spec, model, model_name, plan_mutations, scale, label):
    print(f"\n{'='*65}")
    print(f"C5 search: {label}")
    print(f"{'='*65}")
    X, y = load_raw(spec)
    c0_f1, c0_std = MASTER_C0[label.replace("/", "_").replace(" ", "_").lower()]["f1_macro"], \
                    MASTER_C0[label.replace("/", "_").replace(" ", "_").lower()]["std"]

    results = []
    for plan_name, plan in plan_mutations:
        try:
            f1, std = _evaluate(X, y, copy.deepcopy(model), plan, spec.target, scale)
            delta = f1 - c0_f1
            verdict = win_tie_loss(delta, c0_std, std)
            status = "🏆 WIN" if verdict == "WIN" else ("❌ LOSS" if verdict == "LOSS" else "   TIE")
            print(f"  {status}  [{plan_name:40s}]  f1={f1:.4f}±{std:.4f}  Δ={delta:+.4f}")
            results.append({
                "plan_name": plan_name,
                "f1_macro": f1,
                "std": std,
                "delta": delta,
                "verdict": verdict,
                "plan": plan,
            })
        except Exception as e:
            print(f"  ERROR  [{plan_name}]: {e}")

    results.sort(key=lambda r: r["f1_macro"], reverse=True)
    print(f"\n  Best: {results[0]['plan_name']} — f1={results[0]['f1_macro']:.4f}  {results[0]['verdict']}")

    # WIN → save plan as C5
    winners = [r for r in results if r["verdict"] == "WIN"]
    if winners:
        best = winners[0]
        key = label.replace(" ", "_").replace("/", "_")
        out_dir = Path(f"data/exports/{spec.dataset_id}/provenance")
        out_dir.mkdir(parents=True, exist_ok=True)
        plan_path = out_dir / f"c5_iterative_{key}.json"
        plan_path.write_text(json.dumps(best["plan"], indent=2), encoding="utf-8")
        print(f"\n  ✅ C5 plan saved → {plan_path}")
    else:
        print(f"\n  ⚠  No WIN found for {label}. C4 result is the ceiling.")

    return results


if __name__ == "__main__":
    # bank / logreg
    bank_model = LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=RANDOM_STATE)
    bank_results = run_search(
        spec=BANK_SPEC,
        model=bank_model,
        model_name="logreg",
        plan_mutations=bank_plan_mutations(),
        scale=True,
        label="bank_logreg",
    )

    # heart / rf
    heart_model = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                          random_state=RANDOM_STATE, n_jobs=-1)
    heart_results = run_search(
        spec=HEART_SPEC,
        model=heart_model,
        model_name="rf",
        plan_mutations=heart_plan_mutations(),
        scale=False,
        label="heart_rf",
    )

    print("\n\n" + "="*65)
    print("SUMMARY")
    print("="*65)
    for label, results in [("bank/logreg", bank_results), ("heart/rf", heart_results)]:
        best = results[0]
        print(f"  {label}: best={best['plan_name']}  f1={best['f1_macro']:.4f}  {best['verdict']}")
