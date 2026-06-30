"""Unit tests for the C4 enforcement guardrails (evaluation/enforce_c4_v3.py).

Thesis-canon codes covered here: G1 (clipping policy), G2 (drop restriction),
G8 (imputation sanity), G9 (encoding validity), G11 (KNN median), G12 (scaler
injection). Thesis G6/G7 are enforced inline in evaluation/prepare_conditions.py
inside the C4 plan-preparation flow and are exercised by the Phase B/C
integration runs rather than unit-tested here.
"""
import copy
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.enforce_c4_v3 import enforce_plan, compute_cardinality, SAFE_IQR_K


def make_df():
    return pd.DataFrame({
        "num_a": [1.0, 2.0, 3.0, None, 5.0],
        "num_b": [10, 20, 30, 40, 50],
        "cat_low": ["x", "y", "x", "y", "x"],
        "target": [0, 1, 0, 1, 0],
    })


def base_plan():
    return {"actions": [
        {"action": "handle_missing", "rationale": "", "target_columns": ["num_a"],
         "params": {"strategy": "impute"}},
        {"action": "encode_categorical_per_column", "rationale": "", "target_columns": [],
         "params": {"column_encodings": {"cat_low": "ordinal", "num_b": "ordinal"},
                    "default_method": "ordinal"}},
        {"action": "clip_outliers", "rationale": "", "target_columns": [],
         "params": {"method": "iqr", "iqr_k": 1.5}},
    ]}


def ctx(downstream, redundant=None):
    return {"downstream_model": downstream,
            "redundant_features": redundant or [], "leakage_cols": []}


def run(downstream, plan=None, redundant=None):
    df = make_df()
    plan = copy.deepcopy(plan or base_plan())
    return enforce_plan(plan, ctx(downstream, redundant), compute_cardinality(df))


def actions(plan):
    return [a["action"] for a in plan["actions"]]


# --- G1: outlier clipping policy ---

def test_g1_clip_removed_for_trees():
    plan, _ = run("RandomForest")
    assert "clip_outliers" not in actions(plan)


def test_g1_clip_k_floor_for_linear():
    plan, _ = run("LogisticRegression")
    clip = [a for a in plan["actions"] if a["action"] == "clip_outliers"]
    assert clip and clip[0]["params"]["iqr_k"] >= SAFE_IQR_K


# --- G2 (thesis) / in-file G3: drop restriction ---

def test_g2_arbitrary_drop_blocked():
    plan = base_plan()
    plan["actions"].append({"action": "drop_column", "rationale": "",
                            "target_columns": ["num_b"], "params": {}})
    out, _ = run("RandomForest", plan=plan)
    drops = [a for a in out["actions"] if a["action"] == "drop_column"]
    dropped = [c for a in drops for c in a.get("target_columns", [])]
    assert "num_b" not in dropped


def test_g2_redundant_drop_allowed():
    plan = base_plan()
    plan["actions"].append({"action": "drop_column", "rationale": "",
                            "target_columns": ["num_b"], "params": {}})
    out, _ = run("RandomForest", plan=plan, redundant=["num_b"])
    drops = [a for a in out["actions"] if a["action"] == "drop_column"]
    dropped = [c for a in drops for c in a.get("target_columns", [])]
    assert "num_b" in dropped


# --- G8: imputation sanity ---

def test_g8_tree_forces_median():
    plan, _ = run("GradientBoosting")
    hm = [a for a in plan["actions"] if a["action"] == "handle_missing"][0]
    assert hm["params"]["strategy"] == "median"


def test_g8_linear_keeps_multivariate_imputation():
    plan, _ = run("Ridge")
    hm = [a for a in plan["actions"] if a["action"] == "handle_missing"][0]
    assert hm["params"]["strategy"] == "impute"


# --- G9: encoding validity ---

def test_g9_numeric_columns_stripped_from_encodings():
    plan, _ = run("RandomForest")
    enc = [a for a in plan["actions"] if a["action"] == "encode_categorical_per_column"][0]
    assert "num_b" not in enc["params"]["column_encodings"]
    assert "cat_low" in enc["params"]["column_encodings"]


# --- G11: KNN classifier median imputation ---

def test_g11_knn_classifier_forces_median():
    plan, _ = run("KNN")
    hm = [a for a in plan["actions"] if a["action"] == "handle_missing"][0]
    assert hm["params"]["strategy"] == "median"


def test_g11_knn_regressor_exempt():
    plan, _ = run("KNN_regression")
    hm = [a for a in plan["actions"] if a["action"] == "handle_missing"][0]
    assert hm["params"]["strategy"] == "impute"


# --- G12: scaler injection ---

def test_g12_scaler_injected_for_linear():
    for ds in ("Ridge", "LogisticRegression"):
        plan, _ = run(ds)
        assert "scale_features" in actions(plan)


def test_g12_no_scaler_for_knn_or_trees():
    for ds in ("KNN", "RandomForest", "GradientBoosting"):
        plan, _ = run(ds)
        assert "scale_features" not in actions(plan)


def test_g12_no_duplicate_scaler():
    plan = base_plan()
    plan["actions"].append({"action": "scale_features", "rationale": "",
                            "target_columns": [], "params": {}})
    out, _ = run("Ridge", plan=plan)
    assert actions(out).count("scale_features") == 1


# --- robustness: unknown downstream model ---

def test_unknown_downstream_no_crash_and_generic_only():
    plan, _ = run("NeuralNetwork")
    acts = actions(plan)
    assert "scale_features" not in acts          # no model-aware injection
    hm = [a for a in plan["actions"] if a["action"] == "handle_missing"][0]
    assert hm["params"]["strategy"] == "impute"  # untouched by family rules
