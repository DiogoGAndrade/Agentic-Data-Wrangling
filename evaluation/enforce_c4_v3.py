"""
C4 Guardrails v3 - Final Enforcement Script.

Applies ALL guardrails to existing C4 plan files in one pass:
  G1.  clip_outliers: Removed for tree-based models
  G2.  bin_numeric: Removed entirely (destroys continuous signal)
  G3.  drop_column: Only allows columns in redundant_features/leakage_cols
  G8.  handle_missing: Trees -> median; Linear -> MICE + cat fill_mode
  G9.  encode_categorical_per_column: numeric cols stripped; cardinality-aware thresholds
  G11. handle_missing for KNN classifiers: force median (MICE/mean distort distance space)
  G12. scale_features: Injected for linear models (Ridge, LogReg) if not already present.

RULE-CODE MAPPING (read this before the defence): the labels above are the file's
historical numbering, kept frozen. The thesis canon (Table 6.2) maps as follows:
  in-file G1            = thesis G1  (outlier clipping policy)
  in-file G2            = bin_numeric removal (part of the Section 3.3 action policy,
                          not a numbered rule in Table 6.2)
  in-file G3            = thesis G2  (drop restriction)
  select_features cap   = thesis G3  (conservative feature selection)
  in-file G8/G9/G11/G12 = thesis G8/G9/G11/G12 (same codes)
  thesis G6 (dimensionality budget) and G7 (completeness injection) are enforced in
  evaluation/prepare_conditions.py under the "GUARDRAIL 6" / "GUARDRAIL 7" comments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# --- Constants ---
HIGH_CARD_THRESHOLD_TREE = 10
HIGH_CARD_THRESHOLD_LINEAR = 200
SAFE_IQR_K = 3.0

TREE_KEYWORDS = ["tree", "forest", "random", "gradient", "gbm", "xgboost", "lightgbm"]
LINEAR_KEYWORDS = ["linear", "logistic", "ridge", "lasso", "knn", "svm", "distance"]

DATASETS = {
    "adult": {"target": "income", "models": ["logreg", "rf", "knn", "gbm"]},
    "diabetes": {"target": "readmitted", "models": ["logreg", "rf", "knn", "gbm"]},
    "student": {"target": "final_result", "models": ["logreg", "rf", "knn", "gbm"]},
    "life_expectancy": {"target": "life_expectancy", "models": ["ridge", "rf", "knn", "gbm"]},
}


def load_context(ctx_path: Path) -> Dict[str, Any]:
    raw = json.loads(ctx_path.read_text(encoding="utf-8"))
    if len(raw) == 1:
        return list(raw.values())[0]
    return raw


def compute_cardinality(df: pd.DataFrame) -> Dict[str, int]:
    result = {}
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            result[col] = int(df[col].nunique())
    return result


def enforce_plan(
    plan: Dict[str, Any],
    user_context: Dict[str, Any],
    cardinality: Dict[str, int],
) -> tuple[Dict[str, Any], List[str]]:
    """Apply all C4 guardrails G1-G12. Returns (modified_plan, changes_log)."""
    downstream = (user_context.get("downstream_model") or "").lower()
    is_tree = any(kw in downstream for kw in TREE_KEYWORDS)
    is_linear = any(kw in downstream for kw in LINEAR_KEYWORDS)
    is_knn = "knn" in downstream
    # G11 applies to KNN classifiers only (not KNN regressors)
    # KNN regressor uses downstream "KNN_regression" which contains "regress"
    is_knn_clf = is_knn and ("regress" not in downstream)

    column_semantics = user_context.get("column_semantics", {})
    redundant = set(user_context.get("redundant_features") or [])
    leakage = set(user_context.get("leakage_cols") or [])
    safe_to_drop = redundant | leakage

    high_card_threshold = HIGH_CARD_THRESHOLD_TREE if is_tree else HIGH_CARD_THRESHOLD_LINEAR

    actions = plan.get("actions", [])
    changes: List[str] = []

    # --- PASS 1: Remove/filter destructive actions ---
    filtered = []
    for action in actions:
        act = action.get("action", "")

        # G1: Remove clip_outliers for tree models
        if act == "clip_outliers" and is_tree:
            changes.append("rm clip_outliers(tree)")
            continue

        # G2: Remove bin_numeric entirely
        if act == "bin_numeric":
            changes.append("rm bin_numeric")
            continue

        # G3: Guard drop_column
        if act == "drop_column":
            target_cols = action.get("target_columns", [])
            safe_cols = [c for c in target_cols if c in safe_to_drop]
            blocked = [c for c in target_cols if c not in safe_to_drop]
            if blocked:
                changes.append(f"block_drop:{blocked}")
                if safe_cols:
                    action["target_columns"] = safe_cols
                    filtered.append(action)
                continue

        filtered.append(action)

    # --- PASS 2: Enforce encoding, clip k, select_features, imputation ---
    for action in filtered:
        act = action.get("action", "")
        params = action.get("params") or {}

        if act == "handle_missing":
            current_strat = params.get("strategy", "impute")

            if is_tree and current_strat in ("impute", "mice", "iterative", "mean"):
                # G8 trees: force median
                params["strategy"] = "median"
                changes.append(f"impute:{current_strat}->median(tree)")
                action["params"] = params

            elif is_knn_clf and current_strat in ("impute", "mice", "iterative", "mean"):
                # G11: KNN classification -> median (MICE/mean distort Euclidean distance space)
                params["strategy"] = "median"
                changes.append(f"impute:{current_strat}->median(knn_clf)")
                action["params"] = params

            elif is_linear and not is_knn:
                # G8 linear (LogReg, Ridge): inject fill_mode for categoricals
                if "cat_missing_strategy" not in params:
                    params["cat_missing_strategy"] = "fill_mode"
                    changes.append("cat_missing:fill_mode(linear)")
                    action["params"] = params

        if act == "encode_categorical_per_column":
            col_enc = params.get("column_encodings", {})

            # G9: Remove numeric columns from column_encodings
            numeric_in_enc = [c for c in col_enc if c not in cardinality]
            if numeric_in_enc:
                for c in numeric_in_enc:
                    del col_enc[c]
                changes.append(f"g9_rm_numeric_enc:{numeric_in_enc}")

            for col, sem in column_semantics.items():
                if col not in cardinality:
                    continue
                sem_lower = str(sem).lower()
                card = cardinality.get(col, 0)

                if "ordinal" in sem_lower or "ordered" in sem_lower:
                    correct = "ordinal"
                elif card > high_card_threshold:
                    correct = "ordinal"
                else:
                    correct = "ordinal" if is_tree else "one_hot"

                old = col_enc.get(col, "unknown")
                if old != correct:
                    changes.append(f"enc:{col}:{old}->{correct}")
                col_enc[col] = correct

            params["column_encodings"] = col_enc
            params["default_method"] = "ordinal" if is_tree else "one_hot"
            action["params"] = params

        elif act == "clip_outliers":
            current_k = float(params.get("iqr_k", 1.5))
            if current_k < SAFE_IQR_K:
                changes.append(f"clip_k:{current_k}->{SAFE_IQR_K}")
                params["iqr_k"] = SAFE_IQR_K
                action["params"] = params

        elif act == "select_features":
            llm_drops = list(params.get("drop_columns", []))
            safe_drops = [c for c in llm_drops if c in redundant]
            removed = [c for c in llm_drops if c not in redundant]
            if removed:
                changes.append(f"block_select:{removed}")
            params["drop_columns"] = safe_drops
            params["variance_threshold"] = 0.0
            params["correlation_threshold"] = 1.0
            action["params"] = params

    # --- G12: Inject scale_features for linear models (Ridge, LogReg) ---
    if is_linear and not is_knn:
        has_scale = any(a.get("action") == "scale_features" for a in filtered)
        if not has_scale:
            filtered.append({
                "action": "scale_features",
                "rationale": (
                    "G12: StandardScaler injected for linear model. "
                    "Normalises feature magnitudes for correct Ridge/LogReg regularisation."
                ),
                "target_columns": [],
                "params": {},
            })
            changes.append("g12_inject:scale_features(linear)")

    plan["actions"] = filtered
    return plan, changes


def main():
    parser = argparse.ArgumentParser(description="C4 Guardrails v3 - Final enforcement.")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    contexts_dir = root / "evaluation" / "c4_contexts"
    data_dir = root / "data" / "exports"

    total_plans = 0
    modified_plans = 0

    for dataset_id, info in DATASETS.items():
        raw_path = data_dir / dataset_id / "c0_raw.csv"
        if not raw_path.exists():
            print(f"[SKIP] {dataset_id}: no data at {raw_path}")
            continue

        df = pd.read_csv(raw_path)
        cardinality = compute_cardinality(df)

        prov_dir = data_dir / dataset_id / "provenance"
        if not prov_dir.exists():
            continue

        for ml_model in info["models"]:
            ctx_file = contexts_dir / f"{dataset_id}_{ml_model}.json"
            if not ctx_file.exists():
                continue

            user_context = load_context(ctx_file)
            pattern = f"c4_expanded_plan_*_{ml_model}.json"

            for plan_path in sorted(prov_dir.glob(pattern)):
                total_plans += 1

                raw_text = plan_path.read_text(encoding="utf-8", errors="replace")
                plan = None
                try:
                    plan = json.loads(raw_text)
                except json.JSONDecodeError:
                    cleaned = raw_text.replace("\x00", "").strip()
                    try:
                        plan = json.loads(cleaned)
                    except json.JSONDecodeError:
                        try:
                            last = cleaned.rindex("}")
                            plan = json.loads(cleaned[: last + 1])
                        except (ValueError, json.JSONDecodeError):
                            tag = plan_path.stem.replace("c4_expanded_plan_", "")
                            print(f"[SKIP] {dataset_id}/{tag}: unfixable JSON")
                            continue

                if plan is None:
                    continue

                enforced, changes = enforce_plan(plan, user_context, cardinality)

                if changes:
                    modified_plans += 1
                    tag = plan_path.stem.replace("c4_expanded_plan_", "")
                    if len(changes) <= 5:
                        print(f"[MOD] {dataset_id}/{tag}: {changes}")
                    else:
                        print(f"[MOD] {dataset_id}/{tag}: {len(changes)} changes")

                    if not args.dry_run:
                        plan_path.write_text(
                            json.dumps(enforced, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )

    verb = "Would modify" if args.dry_run else "Modified"
    print(f"\n[DONE] {verb} {modified_plans}/{total_plans} plans.")
    if args.dry_run:
        print("[INFO] Run without --dry-run to apply.")


if __name__ == "__main__":
    main()
