"""
Re-apply C4 enforcement guardrails to existing C4 plans WITHOUT re-calling the LLM.

This script reads existing c4_expanded_plan_*.json files, applies the updated
enforcement logic (high-cardinality check, clip_outliers k floor, bin_numeric removal,
select_features restriction), and overwrites the plan files.

Usage:
    python -m evaluation.reapply_c4_enforcement --root .
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd

# Same constants as in prepare_conditions.py enforcement
# For trees: ordinal is always correct (threshold is moot).
# For linear/distance: regularization handles up to ~200 one-hot cols.
HIGH_CARD_THRESHOLD_TREE = 10
HIGH_CARD_THRESHOLD_LINEAR = 200
SAFE_IQR_K = 3.0

TREE_KEYWORDS = ["tree", "forest", "random", "gradient", "gbm", "xgboost", "lightgbm"]
LINEAR_KEYWORDS = ["linear", "logistic", "ridge", "lasso", "knn", "svm", "distance"]


def load_context(ctx_path: Path) -> Dict[str, Any]:
    """Load a C4 context JSON and extract the inner dataset dict."""
    raw = json.loads(ctx_path.read_text(encoding="utf-8"))
    # Context files wrap everything under the dataset name key
    if len(raw) == 1:
        return list(raw.values())[0]
    return raw


def compute_cardinality(df: pd.DataFrame) -> Dict[str, int]:
    """Compute number of unique values for each string/object column."""
    result = {}
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            result[col] = int(df[col].nunique())
    return result


def enforce_plan(plan: Dict[str, Any], user_context: Dict[str, Any],
                 cardinality: Dict[str, int]) -> Dict[str, Any]:
    """Apply all C4 guardrails to a plan dict. Returns modified plan."""
    downstream = (user_context.get("downstream_model") or "").lower()
    is_tree = any(kw in downstream for kw in TREE_KEYWORDS)
    is_linear = any(kw in downstream for kw in LINEAR_KEYWORDS)

    column_semantics = user_context.get("column_semantics", {})
    allowed_drops: Set[str] = set(user_context.get("redundant_features", []))

    actions = plan.get("actions", [])
    changes_log: List[str] = []

    # Also gather leakage_cols from context (always safe to drop)
    leakage_cols: Set[str] = set(user_context.get("leakage_cols", []))
    safe_to_drop: Set[str] = allowed_drops | leakage_cols

    # PASS 1: Remove/filter destructive actions
    filtered_actions = []
    for action in actions:
        act_name = action.get("action", "")

        # GUARDRAIL: Remove clip_outliers for tree-based models
        if act_name == "clip_outliers" and is_tree:
            changes_log.append("Removed clip_outliers (tree model)")
            continue

        # GUARDRAIL: Remove bin_numeric entirely
        if act_name == "bin_numeric":
            changes_log.append("Removed bin_numeric (signal loss risk)")
            continue

        # GUARDRAIL v3: Restrict drop_column to redundant/leakage features only
        if act_name == "drop_column":
            target_cols = action.get("target_columns", [])
            safe_cols = [c for c in target_cols if c in safe_to_drop]
            blocked_cols = [c for c in target_cols if c not in safe_to_drop]
            if blocked_cols:
                changes_log.append(f"Blocked drop_column: {blocked_cols}")
                if safe_cols:
                    action["target_columns"] = safe_cols
                    rationale = action.get("rationale", "")
                    if "blocked" not in rationale:
                        action["rationale"] = rationale + f" [ENFORCED: blocked {len(blocked_cols)} non-redundant drops]"
                    filtered_actions.append(action)
                # If no safe cols remain, skip the entire action
                continue
            # All cols are safe — keep as-is

        filtered_actions.append(action)

    # PASS 2: Enforce encoding, clip k, select_features
    for action in filtered_actions:
        act_name = action.get("action", "")
        params = action.get("params") or {}

        if act_name == "encode_categorical_per_column":
            col_enc = params.get("column_encodings", {})

            high_card_threshold = HIGH_CARD_THRESHOLD_TREE if is_tree else HIGH_CARD_THRESHOLD_LINEAR

            for col, sem in column_semantics.items():
                sem_lower = str(sem).lower()
                is_high_card = (
                    "high cardinality" in sem_lower
                    or cardinality.get(col, 0) > high_card_threshold
                )

                if "ordinal" in sem_lower or "ordered" in sem_lower:
                    correct = "ordinal"
                elif is_high_card:
                    correct = "ordinal"
                    if not is_tree and col_enc.get(col) != "ordinal":
                        changes_log.append(f"{col}: one_hot->ordinal (card={cardinality.get(col, '?')})")
                else:
                    correct = "ordinal" if is_tree else "one_hot"

                old = col_enc.get(col, "unknown")
                if old != correct:
                    changes_log.append(f"Encoding {col}: {old}->{correct}")
                col_enc[col] = correct

            params["column_encodings"] = col_enc
            params["default_method"] = "ordinal" if is_tree else "one_hot"
            action["params"] = params

            if any("Encoding" in c or "one_hot->ordinal" in c for c in changes_log):
                rationale = action.get("rationale", "")
                if "[ENFORCED" not in rationale:
                    action["rationale"] = rationale + " [ENFORCED: v2 guardrails applied]"

        elif act_name == "encode_categorical":
            correct_method = "ordinal" if is_tree else "one_hot"
            current = params.get("method", "unknown")
            if current != correct_method:
                changes_log.append(f"Global encoding: {current}->{correct_method}")
                params["method"] = correct_method
                action["params"] = params

        elif act_name == "clip_outliers":
            # Only reaches here for linear/distance models
            current_k = float(params.get("iqr_k", 1.5))
            if current_k < SAFE_IQR_K:
                changes_log.append(f"clip_outliers k: {current_k}->{SAFE_IQR_K}")
                params["iqr_k"] = SAFE_IQR_K
                action["params"] = params
                rationale = action.get("rationale", "")
                if "k raised" not in rationale:
                    action["rationale"] = rationale + f" [ENFORCED: k raised to {SAFE_IQR_K}]"

        elif act_name == "select_features":
            llm_drops = list(params.get("drop_columns", []))
            safe_drops = [c for c in llm_drops if c in allowed_drops]
            removed = [c for c in llm_drops if c not in allowed_drops]
            params["drop_columns"] = safe_drops
            params["variance_threshold"] = 0.0
            params["correlation_threshold"] = 1.0
            action["params"] = params
            if removed:
                changes_log.append(f"Blocked drops: {removed}")
                rationale = action.get("rationale", "")
                if "blocked" not in rationale:
                    action["rationale"] = rationale + f" [ENFORCED: blocked {len(removed)} non-redundant drops]"

    plan["actions"] = filtered_actions
    return plan, changes_log


def main():
    parser = argparse.ArgumentParser(description="Re-apply C4 enforcement to existing plans.")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    contexts_dir = root / "evaluation" / "c4_contexts"
    data_dir = root / "data" / "exports"

    DATASETS = {
        "adult": "income",
        "diabetes": "readmitted",
        "student": "final_result",
        "life_expectancy": "life_expectancy",
    }

    ML_MODELS = {
        "adult": ["logreg", "rf", "knn", "gbm"],
        "diabetes": ["logreg", "rf", "knn", "gbm"],
        "student": ["logreg", "rf", "knn", "gbm"],
        "life_expectancy": ["ridge", "rf", "knn", "gbm"],
    }

    total_plans = 0
    modified_plans = 0

    for dataset_id, target in DATASETS.items():
        # Load raw data for cardinality check
        raw_path = data_dir / dataset_id / "c0_raw.csv"
        if not raw_path.exists():
            print(f"[SKIP] {dataset_id}: no raw data at {raw_path}")
            continue

        df = pd.read_csv(raw_path)
        cardinality = compute_cardinality(df)

        prov_dir = data_dir / dataset_id / "provenance"
        if not prov_dir.exists():
            continue

        for ml_model in ML_MODELS.get(dataset_id, []):
            ctx_file = contexts_dir / f"{dataset_id}_{ml_model}.json"
            if not ctx_file.exists():
                continue

            user_context = load_context(ctx_file)

            # Find all C4 plans for this dataset/ml_model
            pattern = f"c4_expanded_plan_*_{ml_model}.json"
            plan_files = sorted(prov_dir.glob(pattern))

            for plan_path in plan_files:
                total_plans += 1

                # Robust JSON loading: handle null bytes and truncation
                raw = plan_path.read_text(encoding="utf-8", errors="replace")
                plan = None
                was_corrupted = False
                try:
                    plan = json.loads(raw)
                except json.JSONDecodeError:
                    was_corrupted = True
                    # Strip null bytes (OneDrive sync artefact)
                    cleaned = raw.replace("\x00", "").strip()
                    try:
                        plan = json.loads(cleaned)
                    except json.JSONDecodeError:
                        # Try parsing up to last valid closing brace
                        try:
                            last_brace = cleaned.rindex("}")
                            plan = json.loads(cleaned[: last_brace + 1])
                        except (ValueError, json.JSONDecodeError):
                            tag = plan_path.stem.replace("c4_expanded_plan_", "")
                            print(f"[SKIP] {dataset_id}/{tag}: unfixable JSON corruption")
                            continue

                if plan is None:
                    continue

                enforced_plan, changes = enforce_plan(plan, user_context, cardinality)

                if was_corrupted:
                    changes.insert(0, "Fixed null-byte corruption")

                if changes or was_corrupted:
                    modified_plans += 1
                    tag = plan_path.stem.replace("c4_expanded_plan_", "")
                    print(f"[MOD] {dataset_id}/{tag}: {', '.join(changes)}")

                    if not args.dry_run:
                        plan_path.write_text(
                            json.dumps(enforced_plan, indent=2, ensure_ascii=False),
                            encoding="utf-8"
                        )

    action = "Would modify" if args.dry_run else "Modified"
    print(f"\n[DONE] {action} {modified_plans}/{total_plans} plans.")
    if args.dry_run:
        print("[INFO] Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
