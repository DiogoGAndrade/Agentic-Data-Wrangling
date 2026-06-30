
# GUARDRAIL CODE MAPPING (thesis Table 6.2 canon vs the inline "GUARDRAIL n" comments
# below, which predate the thesis text and are kept frozen):
#   thesis G1 (outlier clipping policy)        -> "GUARDRAIL 2" and "GUARDRAIL 2b"
#   thesis G2 (drop restriction)               -> "GUARDRAIL 5"
#   thesis G3 (conservative feature selection) -> "GUARDRAIL 3"
#   thesis G6 (dimensionality budget)          -> "GUARDRAIL 6" (plus "GUARDRAIL 1a",
#                                                 high-cardinality ordinal demotion)
#   thesis G7 (completeness injection)         -> "GUARDRAIL 7"
#   thesis G8, G9, G11, G12                    -> evaluation/enforce_c4_v3.py (same codes)
#   "GUARDRAIL 4" (bin_numeric removal)        -> Section 3.3 action policy, not a
#                                                 numbered Table 6.2 rule
# evaluation/prepare_conditions.py
# Code comments in English (per your preference)

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import time

from engine.actions import fix_column_names, handle_missing, deduplicate, normalize_text
from engine.provenance import ProvenanceLog, log_step
from engine.profile_dataset import build_dataset_profile

from llm.ollama_client import OllamaClient
from llm.prompt_templates import (SYSTEM_PROMPT, SYSTEM_PROMPT_C3, SYSTEM_PROMPT_C4,
                                   build_plan_prompt, build_plan_prompt_c3, build_plan_prompt_c4)
from llm.json_utils import extract_json
from engine.schemas import LLMPlan


# -------------------------
# Dataset specs
# -------------------------
DATASET_SPECS: Dict[str, Dict[str, Any]] = {
    # Phase A - model selection (3 classification + 1 regression)
    "adult":            {"target": "income",          "leakage_cols": []},
    "diabetes":         {"target": "readmitted",      "leakage_cols": ["encounter_id", "patient_nbr"]},
    "student":          {"target": "final_result",    "leakage_cols": ["id_student"]},
    "life_expectancy":  {"target": "life_expectancy", "leakage_cols": []},
    # Phase B - held-out generalisation
    "house_prices":     {"target": "SalePrice",       "leakage_cols": ["Id"]},
    "heart":            {"target": "target",          "leakage_cols": []},
    "bank":             {"target": "y",               "leakage_cols": []},
}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _prov_add_step(prov, step: dict) -> None:
    prov.steps.append(step)


def _prov_to_dict(prov: ProvenanceLog) -> dict:
    if hasattr(prov, "to_dict") and callable(getattr(prov, "to_dict")):
        return prov.to_dict()
    if hasattr(prov, "steps") and isinstance(getattr(prov, "steps"), list):
        return {"steps": prov.steps}
    return {"steps": []}


def _string_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            cols.append(c)
    return cols


def _is_small_model(model_tag: str) -> bool:
    """Heuristic for the 'Profile Hiding Optimisation' switch.

    Small models (<=3B params) benefit from aggressive filtering because their
    context window and attention quality degrade quickly under noise.
    Larger models (>=7B) can see the full profile.
    """
    tag = model_tag.lower()
    small_markers = ["1b", "1.5b", "2b", "3b", "3.2b", ":3b"]
    large_markers = ["7b", "8b", "9b", "12b", "13b", "14b", "70b", "nemo"]
    if any(m in tag for m in large_markers):
        return False
    if any(m in tag for m in small_markers):
        return True
    # Default: keep the conservative behaviour (filter ON) for unknown tags.
    return True


# -------------------------
# C1: deterministic baseline (fixed, same for all datasets)
# -------------------------
def apply_c1_pipeline(df: pd.DataFrame, dataset_id: str) -> Tuple[pd.DataFrame, ProvenanceLog]:
    prov = ProvenanceLog()
    df_current = df.copy()

    # C1.1 fix_column_names
    df_after, diff, warnings = fix_column_names(df_current, params={})
    _prov_add_step(
        prov,
        log_step(
            dataset_id=dataset_id,
            step_id=1,
            action_name="fix_column_names",
            approved=True,
            status="applied",
            params={},
            rationale="C1 baseline: standardize column names",
            before_schema={c: str(df_current[c].dtype) for c in df_current.columns},
            after_schema={c: str(df_after[c].dtype) for c in df_after.columns},
            diff_summary=diff,
            warnings=warnings,
            error=None,
        ),
    )
    df_current = df_after

    # C1.2 handle_missing (safe deterministic imputation)
    params_missing = {
        "strategy": "impute",
        "columns": list(df_current.columns[df_current.isna().any()]),
        "impute": {
            "numeric": "median",
            "categorical": "most_frequent",
            "constant_value": 0,
            "constant_categorical": "MISSING",
        },
    }
    df_after, diff, warnings = handle_missing(df_current, params_missing)
    _prov_add_step(
        prov,
        log_step(
            dataset_id=dataset_id,
            step_id=2,
            action_name="handle_missing",
            approved=True,
            status="applied",
            params=params_missing,
            rationale="C1 baseline: deterministic safe imputation",
            before_schema={c: str(df_current[c].dtype) for c in df_current.columns},
            after_schema={c: str(df_after[c].dtype) for c in df_after.columns},
            diff_summary=diff,
            warnings=warnings,
            error=None,
        ),
    )
    df_current = df_after

    # C1.3 deduplicate
    params_dedupe = {"subset": None, "keep": "first", "case_insensitive": False}
    df_after, diff, warnings = deduplicate(df_current, params_dedupe)
    _prov_add_step(
        prov,
        log_step(
            dataset_id=dataset_id,
            step_id=3,
            action_name="deduplicate",
            approved=True,
            status="applied",
            params=params_dedupe,
            rationale="C1 baseline: remove exact duplicate rows",
            before_schema={c: str(df_current[c].dtype) for c in df_current.columns},
            after_schema={c: str(df_after[c].dtype) for c in df_after.columns},
            diff_summary=diff,
            warnings=warnings,
            error=None,
        ),
    )
    df_current = df_after

    # C1.4 normalize_text
    text_cols = _string_columns(df_current)
    if text_cols:
        params_text = {"columns": text_cols, "ops": ["strip", "collapse_whitespace"]}
        df_after, diff, warnings = normalize_text(df_current, params_text)
        _prov_add_step(
            prov,
            log_step(
                dataset_id=dataset_id,
                step_id=4,
                action_name="normalize_text",
                approved=True,
                status="applied",
                params=params_text,
                rationale="C1 baseline: light text normalization",
                before_schema={c: str(df_current[c].dtype) for c in df_current.columns},
                after_schema={c: str(df_after[c].dtype) for c in df_after.columns},
                diff_summary=diff,
                warnings=warnings,
                error=None,
            ),
        )
        df_current = df_after
    else:
        _prov_add_step(
            prov,
            log_step(
                dataset_id=dataset_id,
                step_id=4,
                action_name="normalize_text",
                approved=True,
                status="skipped",
                params={"columns": [], "ops": ["strip", "collapse_whitespace"]},
                rationale="C1 baseline: no string columns",
                before_schema={c: str(df_current[c].dtype) for c in df_current.columns},
                after_schema=None,
                diff_summary=None,
                warnings=["No string columns found; normalize_text skipped."],
                error=None,
            ),
        )

    return df_current, prov


# -------------------------
# C2 helpers: sanitize LLM JSON so Pydantic doesn't explode
# -------------------------
def _as_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _sanitize_plan_obj(obj: Any) -> Dict[str, Any]:
    """
    Makes the model output conform to the expected LLMPlan schema as much as possible.
    This version also maps 'transformations_to_consider' into executable action params.
    """
    if not isinstance(obj, dict):
        obj = {}

    obj.setdefault("dataset_summary", "")
    obj.setdefault("diagnostics", [])
    obj.setdefault("prognostics", [])
    obj.setdefault("actions", [])
    obj.setdefault("assumptions", [])

    if not isinstance(obj.get("dataset_summary"), str):
        obj["dataset_summary"] = str(obj.get("dataset_summary", ""))

    # -------------------------
    # diagnostics
    # -------------------------
    diags = []
    for d in _as_list(obj.get("diagnostics")):
        if isinstance(d, str):
            diags.append({"code": "note", "severity": "info", "message": d, "evidence": {}})
            continue
        if not isinstance(d, dict):
            continue

        code = d.get("code") or "unknown"
        severity = d.get("severity") or "info"
        message = d.get("message") or d.get("observation") or "No message."
        evidence = d.get("evidence") or {}

        diags.append({
            "code": str(code),
            "severity": str(severity),
            "message": str(message),
            "evidence": evidence,
        })

    obj["diagnostics"] = diags

    # -------------------------
    # prognostics
    # -------------------------
    progs = []
    for p in _as_list(obj.get("prognostics")):
        if isinstance(p, str):
            progs.append({
                "code": p,
                "severity": "info",
                "message": "Flag/check suggested by LLM.",
                "evidence": {},
            })
            continue
        if not isinstance(p, dict):
            continue

        code = p.get("code") or p.get("name") or "unknown"
        severity = p.get("severity") or "info"
        message = p.get("message") or p.get("observation") or "No message."
        evidence = p.get("evidence") or {}

        progs.append({
            "code": str(code),
            "severity": str(severity),
            "message": str(message),
            "evidence": evidence,
        })

    obj["prognostics"] = progs

    # -------------------------
    # transformations_to_consider map
    # -------------------------
    ttc_map = {}
    for t in _as_list(obj.get("transformations_to_consider")):
        if not isinstance(t, dict):
            continue
        t_name = t.get("name")
        if isinstance(t_name, str):
            ttc_map[t_name] = t

    # -------------------------
    # actions
    # -------------------------
    actions = []
    for a in _as_list(obj.get("actions")):
        if isinstance(a, str):
            continue
        if not isinstance(a, dict):
            continue

        action = a.get("action") or a.get("name") or a.get("code")
        if action is None:
            continue
        action = str(action)

        rationale = a.get("rationale") or a.get("description") or a.get("message") or "No rationale."

        # Extract target_columns (fallback to empty list if missing)
        target_columns = a.get("target_columns")
        if not isinstance(target_columns, list):
            target_columns = []

        raw_params = a.get("params") or {}
        if not isinstance(raw_params, dict):
            raw_params = {}

        actions.append({
            "action": action,
            "rationale": str(rationale),
            "target_columns": target_columns,
            "params": raw_params,
        })

    obj["actions"] = actions

    # -------------------------
    # assumptions
    # -------------------------
    ass = _as_list(obj.get("assumptions"))
    obj["assumptions"] = [str(x) for x in ass if x is not None]

    return obj

# -------------------------
# C2: LLM-assisted (offline execution for experiments)
# -------------------------
def apply_c2_llm(
    df: pd.DataFrame,
    dataset_id: str,
    target_column: str,
    ollama_url: str,
    ollama_model: str,
    debug_dir: Optional[Path] = None,
    user_context: Optional[Dict[str, Any]] = None,
    condition_mode: str = "c2",
) -> Tuple[pd.DataFrame, ProvenanceLog, LLMPlan]:
    prov = ProvenanceLog()
    df_current = df.copy()

    columns = list(df_current.columns)
    preview_rows = df_current.head(8).to_dict(orient="records")
    dataset_profile = build_dataset_profile(df_current, target_column=target_column)

    client = OllamaClient(base_url=ollama_url, model=ollama_model)

    # Heuristic: small models (<=3B) get the "Profile Hiding" filter to protect
    # token budget; larger models (>=7B) see the full profile.
    aggressive = _is_small_model(ollama_model)

    if condition_mode == "c4" and user_context is not None:
        prompt = build_plan_prompt_c4(
            dataset_name=dataset_id,
            columns=columns,
            preview_rows=preview_rows,
            target_column=target_column,
            dataset_profile=dataset_profile,
            aggressive_filter=False,  # C4: show ALL columns
            user_context=user_context,
        )
    elif condition_mode == "c3" and user_context is not None:
        prompt = build_plan_prompt_c3(
            dataset_name=dataset_id,
            columns=columns,
            preview_rows=preview_rows,
            target_column=target_column,
            dataset_profile=dataset_profile,
            aggressive_filter=aggressive,
            user_context=user_context,
        )
    else:
        prompt = build_plan_prompt(
            dataset_name=dataset_id,
            columns=columns,
            preview_rows=preview_rows,
            target_column=target_column,
            dataset_profile=dataset_profile,
            aggressive_filter=aggressive,
        )

    if debug_dir is not None:
        _ensure_dir(debug_dir)
        (debug_dir / f"{dataset_id}_dataset_profile.json").write_text(
            json.dumps(dataset_profile, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _write_debug(name: str, content: str) -> None:
        if debug_dir is not None:
            (debug_dir / f"{dataset_id}_{name}").write_text(content, encoding="utf-8")

    def _is_useful_plan(plan_obj: LLMPlan) -> bool:
        if not hasattr(plan_obj, "actions") or len(plan_obj.actions) == 0:
            return False
        action_names = [a.action.value for a in plan_obj.actions]
        if action_names == ["fix_column_names"]:
            return False
        return True

    def _build_fallback_plan() -> LLMPlan:
        """Safe fallback plan when the LLM fails to produce valid JSON after all retries."""
        encoding_method = "one_hot"
        is_tree = False
        if condition_mode in ("c3", "c4") and user_context is not None:
            downstream = (user_context.get("downstream_model") or "").lower()
            tree_kw = ["tree", "forest", "random", "gradient", "gbm", "xgboost", "lightgbm"]
            is_tree = any(kw in downstream for kw in tree_kw)
            if is_tree:
                encoding_method = "ordinal"

        if condition_mode == "c4" and user_context is not None:
            column_semantics = user_context.get("column_semantics", {})
            column_encodings = {}
            for col, sem in column_semantics.items():
                sem_lower = str(sem).lower()
                if "ordinal" in sem_lower or "ordered" in sem_lower:
                    column_encodings[col] = "ordinal"
                else:
                    column_encodings[col] = "ordinal" if is_tree else "one_hot"

            encode_action = {
                "action": "encode_categorical_per_column",
                "rationale": f"Fallback: per-column encoding from context (default={encoding_method}).",
                "target_columns": [],
                "params": {"column_encodings": column_encodings, "default_method": encoding_method}
            }
        else:
            encode_action = {
                "action": "encode_categorical",
                "rationale": f"Fallback: {encoding_method} encoding based on downstream model context.",
                "target_columns": [],
                "params": {"method": encoding_method}
            }

        fallback_dict = {
            "reasoning_steps": ["LLM generation failed after 3 attempts. Applying context-aware fallback."],
            "dataset_summary": "Fallback plan due to parsing failure.",
            "diagnostics": [],
            "prognostics": [],
            "actions": [
                {"action": "fix_column_names", "rationale": "Fallback: standardize column names.",
                 "target_columns": [], "params": {}},
                {"action": "handle_missing", "rationale": "Fallback: impute missing values.",
                 "target_columns": [], "params": {"strategy": "impute"}},
                {"action": "normalize_text", "rationale": "Fallback: standardize strings.",
                 "target_columns": [], "params": {}},
                encode_action,
            ],
            "assumptions": ["LLM failed to produce a valid plan; encoding method inferred from user context."]
        }
        return LLMPlan.model_validate(fallback_dict)

    plan: Optional[LLMPlan] = None
    last_error: Optional[str] = None

    # Try up to 3 times to get a valid/useful plan
    for attempt in range(1, 4):
        extra_instruction = ""
        if attempt > 1:
            extra_instruction = (
                "\n\nSTRICT RETRY INSTRUCTIONS:\n"
                "- Return ONLY valid JSON.\n"
                "- actions MUST be a non-empty list.\n"
                "- Each action item MUST contain exactly: action, rationale, params.\n"
                "- If uncertain, include at least fix_column_names with empty params {}.\n"
                "- If text columns exist, include normalize_text with strip and collapse_whitespace.\n"
                "- Do NOT return only diagnostics/prognostics.\n"
            )

        if condition_mode == "c4":
            system_prompt = SYSTEM_PROMPT_C4
        elif condition_mode == "c3":
            system_prompt = SYSTEM_PROMPT_C3
        else:
            system_prompt = SYSTEM_PROMPT
        raw = client.generate(
            prompt=prompt + extra_instruction,
            system=system_prompt,
            temperature=0.2,
        )

        _write_debug(f"raw_llm_attempt_{attempt}.txt", raw)

        try:
            obj = extract_json(raw)
            if isinstance(obj, dict):
                obj["_columns_runtime"] = columns
            obj = _sanitize_plan_obj(obj)
            _write_debug(
                f"plan_sanitized_attempt_{attempt}.json",
                json.dumps(obj, indent=2, ensure_ascii=False),
            )

            plan_candidate = LLMPlan.model_validate(obj)

            if _is_useful_plan(plan_candidate):
                plan = plan_candidate
                break
            else:
                last_error = f"Attempt {attempt}: valid schema but empty/no-op actions."
        except Exception as e:
            last_error = f"Attempt {attempt}: {type(e).__name__}: {e}"

    # If all retries failed or produced empty plans, force deterministic fallback
    if plan is None:
        _prov_add_step(
            prov,
            log_step(
                dataset_id=dataset_id,
                step_id=0,
                action_name="llm_plan_parse",
                approved=True,
                status="failed",
                params={},
                rationale="LLM output was invalid or produced no executable actions; deterministic fallback plan applied.",
                before_schema={c: str(df_current[c].dtype) for c in df_current.columns},
                after_schema=None,
                diff_summary=None,
                warnings=None,
                error=last_error,
            ),
        )
        plan = _build_fallback_plan()

    # -------------------------------------------------------------------------
    # CONTEXT ENFORCEMENT (post-LLM correction)
    # -------------------------------------------------------------------------
    if condition_mode in ("c3", "c4") and user_context is not None:
        downstream = (user_context.get("downstream_model") or "").lower()
        tree_keywords = ["tree", "forest", "random", "gradient", "gbm", "xgboost", "lightgbm"]
        linear_keywords = ["linear", "logistic", "ridge", "lasso", "knn", "svm", "distance"]

        is_tree = any(kw in downstream for kw in tree_keywords)
        is_linear = any(kw in downstream for kw in linear_keywords)
        is_knn = "knn" in downstream or "nearest" in downstream

        if condition_mode == "c4":
            # ----- C4 ENFORCEMENT: per-column encoding -----
            column_semantics = user_context.get("column_semantics", {})
            # High-cardinality threshold: force ordinal above this.
            # For trees: all encoding is ordinal anyway (threshold is moot).
            # For linear/distance: regularization handles up to ~200 one-hot cols;
            # only force ordinal for extreme cardinality (diag_1=717 in diabetes).
            HIGH_CARD_THRESHOLD = 200 if is_linear else 10
            # Safe minimum IQR k for clip_outliers
            SAFE_IQR_K = 3.0
            # Allowed redundant features from user context
            allowed_drops = set(user_context.get("redundant_features", []))
            leakage_cols = set(user_context.get("leakage_cols", []))
            safe_to_drop = allowed_drops | leakage_cols

            # Compute actual cardinality from the data for robust enforcement
            _actual_cardinality = {}
            for col in df_current.columns:
                if pd.api.types.is_object_dtype(df_current[col]) or pd.api.types.is_string_dtype(df_current[col]):
                    _actual_cardinality[col] = int(df_current[col].nunique())

            # First pass: filter out destructive actions
            actions_to_remove = []
            for i, action in enumerate(plan.actions):
                # GUARDRAIL 2: clip_outliers - remove for tree-based models
                if action.action.value == "clip_outliers" and is_tree:
                    actions_to_remove.append(i)
                    print(f"       [C4 ENFORCE] Removed clip_outliers (trees are robust to outliers)")
                # GUARDRAIL 4: bin_numeric - remove entirely (destroys signal)
                elif action.action.value == "bin_numeric":
                    actions_to_remove.append(i)
                    print(f"       [C4 ENFORCE] Removed bin_numeric (signal loss risk)")
                # GUARDRAIL 5: drop_column - only allow redundant/leakage columns
                elif action.action.value == "drop_column":
                    target_cols = action.target_columns or []
                    safe_cols = [c for c in target_cols if c in safe_to_drop]
                    blocked_cols = [c for c in target_cols if c not in safe_to_drop]
                    if blocked_cols:
                        print(f"       [C4 ENFORCE] Blocked drop_column: {blocked_cols}")
                        if safe_cols:
                            action.target_columns = safe_cols
                        else:
                            actions_to_remove.append(i)

            # Remove in reverse order to preserve indices
            for i in reversed(actions_to_remove):
                plan.actions.pop(i)

            # Second pass: enforce encoding, clip_outliers k, select_features
            for action in plan.actions:
                if action.action.value == "encode_categorical_per_column":
                    if action.params is None:
                        action.params = {}
                    col_enc = action.params.get("column_encodings", {})

                    enforced = {}
                    for col, sem in column_semantics.items():
                        sem_lower = str(sem).lower()
                        # Check for high-cardinality: from semantics tag OR actual data
                        is_high_card = (
                            "high cardinality" in sem_lower
                            or _actual_cardinality.get(col, 0) > HIGH_CARD_THRESHOLD
                        )
                        if "ordinal" in sem_lower or "ordered" in sem_lower:
                            enforced[col] = "ordinal"
                        elif is_high_card:
                            # GUARDRAIL 1a: high-cardinality nominals always use ordinal
                            # to avoid dimensionality explosion with one_hot
                            enforced[col] = "ordinal"
                            if not is_tree:
                                card = _actual_cardinality.get(col, "?")
                                print(f"       [C4 ENFORCE] {col}: one_hot->ordinal (cardinality={card} > {HIGH_CARD_THRESHOLD})")
                        else:
                            enforced[col] = "ordinal" if is_tree else "one_hot"

                    # ---------------------------------------------------------------
                    # GUARDRAIL 6: Dimensionality budget for distance/linear models
                    # Estimate total post-encoding feature count.  If one_hot would
                    # push the total beyond MAX_FEATURES, progressively demote the
                    # highest-cardinality one_hot columns to ordinal until the budget
                    # is met.  This prevents curse-of-dimensionality degradation for
                    # KNN, Logistic Regression, Ridge, etc.
                    # ---------------------------------------------------------------
                    is_knn = "knn" in downstream or "nearest" in downstream
                    if is_linear:
                        # KNN: sparse binary features degrade distance metrics severely.
                        # Linear models (logreg, ridge) tolerate more via regularization.
                        MAX_FEATURES = 50 if is_knn else 200
                        n_numeric = sum(
                            1 for c in df_current.columns
                            if pd.api.types.is_numeric_dtype(df_current[c]) and c != target_column
                        )
                        # Count how many one_hot features we'd create
                        onehot_cols_planned = [
                            col for col, method in enforced.items()
                            if method == "one_hot" and col in _actual_cardinality
                        ]
                        # Also count one_hot from cat columns NOT in enforced (using default)
                        all_cat = set(_actual_cardinality.keys())
                        default_method = "ordinal" if is_tree else "one_hot"
                        for cat_col in all_cat:
                            if cat_col not in enforced and cat_col != target_column:
                                if default_method == "one_hot":
                                    onehot_cols_planned.append(cat_col)
                                    enforced[cat_col] = "one_hot"

                        onehot_feature_count = sum(
                            _actual_cardinality.get(c, 2)
                            for c in onehot_cols_planned
                        )
                        ordinal_count = sum(
                            1 for col, method in enforced.items()
                            if method == "ordinal"
                        )
                        estimated_total = n_numeric + onehot_feature_count + ordinal_count

                        if estimated_total > MAX_FEATURES:
                            print(f"       [C4 ENFORCE] Dimensionality budget: estimated {estimated_total} features "
                                  f"(max {MAX_FEATURES}). Demoting one_hot columns to ordinal.")
                            # Sort one_hot columns by cardinality descending - demote biggest first
                            onehot_by_card = sorted(
                                onehot_cols_planned,
                                key=lambda c: _actual_cardinality.get(c, 0),
                                reverse=True,
                            )
                            for col in onehot_by_card:
                                if estimated_total <= MAX_FEATURES:
                                    break
                                card = _actual_cardinality.get(col, 2)
                                # Demoting from one_hot (card features) to ordinal (1 feature)
                                estimated_total -= (card - 1)
                                enforced[col] = "ordinal"
                                print(f"       [C4 ENFORCE] {col}: one_hot->ordinal "
                                      f"(dimensionality budget, card={card}, est_total now {estimated_total})")
                    # ---------------------------------------------------------------

                    changes = []
                    for col, method in enforced.items():
                        old = col_enc.get(col, "unknown")
                        if old != method:
                            changes.append(f"{col}: {old}->{method}")
                        col_enc[col] = method

                    action.params["column_encodings"] = col_enc
                    default = "ordinal" if is_tree else "one_hot"
                    # If dimensionality budget forced demotions, override default too
                    if is_linear and estimated_total > MAX_FEATURES:
                        default = "ordinal"
                        print(f"       [C4 ENFORCE] Default encoding overridden to ordinal (dimensionality budget)")
                    action.params["default_method"] = default

                    if changes:
                        print(f"       [C4 ENFORCE] Per-column overrides ({len(changes)}): "
                              + "; ".join(changes[:5]) + ("..." if len(changes) > 5 else ""))
                        action.rationale += f" [ENFORCED: {len(changes)} columns corrected for {downstream}]"

                elif action.action.value == "encode_categorical":
                    correct_method = "ordinal" if is_tree else "one_hot"
                    current_method = (action.params or {}).get("method", "unknown")
                    if current_method != correct_method:
                        print(f"       [C4 ENFORCE] Global encoding override: {current_method} -> {correct_method}")
                        if action.params is None:
                            action.params = {}
                        action.params["method"] = correct_method
                        action.rationale += f" [ENFORCED: {correct_method} for {downstream}]"

                # GUARDRAIL 2b: clip_outliers for linear - enforce safe k
                elif action.action.value == "clip_outliers":
                    if action.params is None:
                        action.params = {}
                    current_k = float(action.params.get("iqr_k", 1.5))
                    if current_k < SAFE_IQR_K:
                        print(f"       [C4 ENFORCE] clip_outliers k: {current_k} -> {SAFE_IQR_K}")
                        action.params["iqr_k"] = SAFE_IQR_K
                        action.rationale += f" [ENFORCED: k raised to {SAFE_IQR_K} for safety]"

                # GUARDRAIL 3: select_features - only allow user-approved drops
                elif action.action.value == "select_features":
                    if action.params is None:
                        action.params = {}
                    llm_drops = list(action.params.get("drop_columns", []))
                    # Only keep drops that are in the user-approved redundant list
                    safe_drops = [c for c in llm_drops if c in allowed_drops]
                    removed_drops = [c for c in llm_drops if c not in allowed_drops]
                    action.params["drop_columns"] = safe_drops
                    # Disable automatic variance/correlation drops - too risky
                    action.params["variance_threshold"] = 0.0
                    action.params["correlation_threshold"] = 1.0
                    if removed_drops:
                        print(f"       [C4 ENFORCE] select_features: blocked drops {removed_drops}")
                        action.rationale += f" [ENFORCED: blocked {len(removed_drops)} non-redundant drops]"

            # ---------------------------------------------------------------
            # GUARDRAIL 7: Completeness - inject encoding if LLM omitted it
            # If the dataset has categorical columns but the plan has no
            # encode_categorical or encode_categorical_per_column action,
            # inject one.  This prevents the LLM's conservatism from leaving
            # string columns untouched (which either errors out or gets
            # silently dropped by sklearn).
            # ---------------------------------------------------------------
            has_encoding_action = any(
                a.action.value in ("encode_categorical", "encode_categorical_per_column")
                for a in plan.actions
            )
            if not has_encoding_action and _actual_cardinality:
                # Build per-column encodings from context
                injected_enc = {}
                for col, sem in column_semantics.items():
                    sem_lower = str(sem).lower()
                    if "ordinal" in sem_lower or "ordered" in sem_lower:
                        injected_enc[col] = "ordinal"
                    else:
                        injected_enc[col] = "ordinal" if is_tree else "one_hot"
                # Also cover cat columns not in semantics
                for col in _actual_cardinality:
                    if col not in injected_enc and col != target_column:
                        injected_enc[col] = "ordinal" if is_tree else "one_hot"

                # Apply dimensionality budget (reuse Guardrail 6 logic)
                if is_linear:
                    MAX_FEATURES_INJ = 50 if is_knn else 200
                    n_numeric_inj = sum(
                        1 for c in df_current.columns
                        if pd.api.types.is_numeric_dtype(df_current[c]) and c != target_column
                    )
                    onehot_inj = [c for c, m in injected_enc.items() if m == "one_hot"]
                    onehot_feat_inj = sum(_actual_cardinality.get(c, 2) for c in onehot_inj)
                    ordinal_inj = sum(1 for c, m in injected_enc.items() if m == "ordinal")
                    est_total_inj = n_numeric_inj + onehot_feat_inj + ordinal_inj
                    if est_total_inj > MAX_FEATURES_INJ:
                        # Demote biggest one_hot columns first
                        for col in sorted(onehot_inj, key=lambda c: _actual_cardinality.get(c, 0), reverse=True):
                            if est_total_inj <= MAX_FEATURES_INJ:
                                break
                            card = _actual_cardinality.get(col, 2)
                            est_total_inj -= (card - 1)
                            injected_enc[col] = "ordinal"

                default_inj = "ordinal" if is_tree else "one_hot"
                # If budget forced many demotions, default to ordinal
                if is_linear:
                    onehot_remaining = sum(1 for m in injected_enc.values() if m == "one_hot")
                    if onehot_remaining == 0:
                        default_inj = "ordinal"

                from engine.schemas import Action as ActionSchema, ActionType
                injected_action = ActionSchema(
                    action=ActionType.encode_categorical_per_column,
                    rationale="[INJECTED] LLM plan omitted encoding for categorical columns. "
                              "Injected context-aware per-column encoding with dimensionality budget.",
                    target_columns=[],
                    params={
                        "column_encodings": injected_enc,
                        "default_method": default_inj,
                    },
                )
                # Insert after handle_missing (if present), otherwise at end
                insert_idx = len(plan.actions)
                for idx, a in enumerate(plan.actions):
                    if a.action.value == "handle_missing":
                        insert_idx = idx + 1
                        break
                plan.actions.insert(insert_idx, injected_action)
                print(f"       [C4 ENFORCE] INJECTED encode_categorical_per_column "
                      f"({len(injected_enc)} cols, default={default_inj})")

        elif is_tree or is_linear:
            # ----- C3 ENFORCEMENT: global encoding override -----
            correct_method = "ordinal" if is_tree else "one_hot"
            for action in plan.actions:
                if action.action.value == "encode_categorical":
                    current_method = (action.params or {}).get("method", "unknown")
                    if current_method != correct_method:
                        print(f"       [C3 ENFORCE] Overriding encoding: {current_method} -> {correct_method} "
                              f"(downstream={downstream})")
                        if action.params is None:
                            action.params = {}
                        action.params["method"] = correct_method
                        action.rationale += f" [ENFORCED: {correct_method} for {downstream}]"

    # Import full set of actions
    from engine.actions import cast_type, drop_column, encode_categorical, remove_outliers

    # ---- C4 standalone executors for prepare_conditions CSV generation ----
    def _exec_encode_categorical_per_column(df_in: pd.DataFrame, params: dict):
        column_encodings = params.get("column_encodings", {})
        default_method = params.get("default_method", "one_hot")
        cat_cols = [c for c in df_in.columns
                    if (pd.api.types.is_object_dtype(df_in[c]) or pd.api.types.is_string_dtype(df_in[c]))
                    and c != target_column]
        if not cat_cols:
            return df_in, {"no_categorical_columns": True}, []
        from sklearn.preprocessing import OrdinalEncoder as OrdEnc, OneHotEncoder as OHEnc
        ordinal_cols = [c for c in cat_cols if column_encodings.get(c, default_method) == "ordinal"]
        onehot_cols = [c for c in cat_cols if column_encodings.get(c, default_method) == "one_hot"]
        warnings_list = []
        if ordinal_cols:
            df_in[ordinal_cols] = df_in[ordinal_cols].astype(str).fillna("MISSING")
            enc = OrdEnc(handle_unknown="use_encoded_value", unknown_value=-1)
            arr = enc.fit_transform(df_in[ordinal_cols])
            new_cols = [f"{c}__ord" for c in ordinal_cols]
            enc_df = pd.DataFrame(arr, columns=new_cols, index=df_in.index)
            df_in = df_in.drop(columns=ordinal_cols).join(enc_df)
            warnings_list.append(f"Ordinal encoded {len(ordinal_cols)} cols")
        if onehot_cols:
            df_in[onehot_cols] = df_in[onehot_cols].astype(str).fillna("MISSING")
            enc = OHEnc(handle_unknown="ignore", sparse_output=False)
            arr = enc.fit_transform(df_in[onehot_cols])
            new_cols = list(enc.get_feature_names_out(onehot_cols))
            enc_df = pd.DataFrame(arr, columns=new_cols, index=df_in.index)
            df_in = df_in.drop(columns=onehot_cols).join(enc_df)
            warnings_list.append(f"One-hot encoded {len(onehot_cols)} cols")
        return df_in, {"ordinal": ordinal_cols, "one_hot": onehot_cols}, warnings_list

    def _exec_select_features(df_in: pd.DataFrame, params: dict):
        import numpy as np
        variance_threshold = float(params.get("variance_threshold", 0.01))
        correlation_threshold = float(params.get("correlation_threshold", 0.95))
        explicit_drops = list(params.get("drop_columns", []))
        dropped = []
        valid_drops = [c for c in explicit_drops if c in df_in.columns and c != target_column]
        if valid_drops:
            df_in = df_in.drop(columns=valid_drops, errors="ignore")
            dropped.extend(valid_drops)
        num_cols = [c for c in df_in.columns
                    if pd.api.types.is_numeric_dtype(df_in[c]) and c != target_column]
        if num_cols and variance_threshold > 0:
            variances = df_in[num_cols].var()
            low_var = [c for c in num_cols if variances.get(c, 1) < variance_threshold]
            if low_var:
                df_in = df_in.drop(columns=low_var, errors="ignore")
                dropped.extend(low_var)
                num_cols = [c for c in num_cols if c not in low_var]
        if len(num_cols) >= 2 and correlation_threshold < 1.0:
            try:
                corr_matrix = df_in[num_cols].corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                corr_drops = [col for col in upper.columns if any(upper[col] > correlation_threshold)]
                if corr_drops:
                    df_in = df_in.drop(columns=corr_drops, errors="ignore")
                    dropped.extend(corr_drops)
            except Exception:
                pass
        return df_in, {"dropped_features": dropped}, [f"Dropped {len(dropped)} features"] if dropped else []

    def _exec_clip_outliers(df_in: pd.DataFrame, params: dict):
        params.setdefault("mode", "clip")
        params.setdefault("iqr_k", 1.5)
        params.setdefault("method", "iqr")
        if "columns" not in params:
            num_cols = [c for c in df_in.columns
                        if pd.api.types.is_numeric_dtype(df_in[c]) and c != target_column]
            params["columns"] = num_cols
        return remove_outliers(df_in, params)

    def _exec_bin_numeric(df_in: pd.DataFrame, params: dict):
        from sklearn.preprocessing import KBinsDiscretizer
        cols = params.get("columns", [])
        cols = [c for c in cols if c in df_in.columns and pd.api.types.is_numeric_dtype(df_in[c])
                and c != target_column]
        if not cols:
            return df_in, {}, ["No valid numeric columns to bin"]
        n_bins = int(params.get("n_bins", 5))
        strategy = params.get("strategy", "quantile")
        encode_bins = params.get("encode_bins", "ordinal")
        for c in cols:
            if df_in[c].isna().any():
                df_in[c] = df_in[c].fillna(df_in[c].median())
        sklearn_encode = "ordinal" if encode_bins == "ordinal" else "onehot-dense"
        try:
            binner = KBinsDiscretizer(n_bins=n_bins, encode=sklearn_encode,
                                      strategy=strategy, subsample=None)
            arr = binner.fit_transform(df_in[cols])
            if encode_bins == "ordinal":
                new_cols = [f"{c}__bin" for c in cols]
            else:
                new_cols = [f"bin_{i}" for i in range(arr.shape[1])]
            enc_df = pd.DataFrame(arr, columns=new_cols, index=df_in.index)
            df_in = df_in.drop(columns=cols).join(enc_df)
        except Exception as e:
            return df_in, {}, [f"Binning failed: {e}"]
        return df_in, {"binned": cols, "new_cols": new_cols}, []

    ACTION_EXECUTORS = {
        "fix_column_names": fix_column_names,
        "cast_type": cast_type,
        "handle_missing": handle_missing,
        "normalize_text": normalize_text,
        "deduplicate": deduplicate,
        "drop_column": drop_column,
        "encode_categorical": encode_categorical,
        "remove_outliers": remove_outliers,
        # C4 actions
        "encode_categorical_per_column": _exec_encode_categorical_per_column,
        "clip_outliers": _exec_clip_outliers,
        "select_features": _exec_select_features,
        "bin_numeric": _exec_bin_numeric,
    }

    for step_id, a in enumerate(plan.actions, start=1):
        action_name = a.action.value
        executor = ACTION_EXECUTORS.get(action_name)

        before_schema = {c: str(df_current[c].dtype) for c in df_current.columns}

        # --- TRANSLATION LAYER: LLM Intent to Python Params ---
        exec_params = {}
        target_cols = a.target_columns
        valid_cols = [c for c in target_cols if c in df_current.columns]
        hallucinated_cols = [c for c in target_cols if c not in df_current.columns]
        if hallucinated_cols:
            # The LLM proposed a column that does not exist in this dataset (a common
            # failure mode is echoing a column name from the few-shot example in the
            # system prompt instead of reasoning about the real data). This is exactly
            # what the guardrail layer is meant to catch: the action below proceeds
            # without these columns, and downstream validation will reject or no-op it
            # if nothing valid remains.
            warnings.warn(
                f"[guardrail] LLM proposed non-existent column(s) {hallucinated_cols} "
                f"for action '{action_name}' on dataset '{dataset_id}'; discarding them."
            )
        if valid_cols and "all" not in [str(c).lower() for c in target_cols]:
            exec_params["columns"] = valid_cols

        if a.params:
            exec_params.update(a.params)

        # Safety injection / parameter completion
        if action_name == "drop_column":
            exec_params.setdefault("target_column", target_column)

        elif action_name == "normalize_text":
            if "columns" not in exec_params:
                text_cols = [c for c in df_current.columns
                             if pd.api.types.is_object_dtype(df_current[c])
                             or pd.api.types.is_string_dtype(df_current[c])]
                exec_params["columns"] = text_cols
            exec_params.setdefault("ops", ["strip", "collapse_whitespace"])

        elif action_name == "handle_missing":
            if "columns" not in exec_params:
                cols_with_missing = [c for c in df_current.columns if df_current[c].isna().sum() > 0]
                exec_params["columns"] = cols_with_missing
            exec_params.setdefault("strategy", "impute")
            exec_params.setdefault("impute", {
                "numeric": "median", "categorical": "most_frequent",
                "constant_value": 0, "constant_categorical": "MISSING"
            })

        elif action_name == "encode_categorical":
            if "columns" not in exec_params:
                cat_cols = [c for c in df_current.columns
                            if (pd.api.types.is_object_dtype(df_current[c])
                                or pd.api.types.is_string_dtype(df_current[c]))
                            and df_current[c].nunique(dropna=True) <= 20
                            and c != target_column]
                exec_params["columns"] = cat_cols

        elif action_name == "remove_outliers":
            if "columns" not in exec_params:
                num_cols = [c for c in df_current.columns
                            if pd.api.types.is_numeric_dtype(df_current[c]) and c != target_column]
                exec_params["columns"] = num_cols
            exec_params.setdefault("method", "iqr")
            exec_params.setdefault("iqr_k", 3.0)
            exec_params.setdefault("mode", "clip")

        elif action_name == "deduplicate":
            exec_params.setdefault("subset", None)
            exec_params.setdefault("keep", "first")
            exec_params.setdefault("case_insensitive", False)

        # --- C4 ACTIONS ---
        elif action_name == "encode_categorical_per_column":
            exec_params.setdefault("column_encodings", {})
            exec_params.setdefault("default_method", "one_hot")

        elif action_name == "clip_outliers":
            if "columns" not in exec_params:
                num_cols = [c for c in df_current.columns
                            if pd.api.types.is_numeric_dtype(df_current[c]) and c != target_column]
                exec_params["columns"] = num_cols
            exec_params.setdefault("method", "iqr")
            exec_params.setdefault("iqr_k", 1.5)
            exec_params.setdefault("mode", "clip")

        elif action_name == "select_features":
            exec_params.setdefault("variance_threshold", 0.01)
            exec_params.setdefault("correlation_threshold", 0.95)
            exec_params.setdefault("drop_columns", [])

        elif action_name == "bin_numeric":
            exec_params.setdefault("n_bins", 5)
            exec_params.setdefault("strategy", "quantile")
            exec_params.setdefault("encode_bins", "ordinal")
        # --- END TRANSLATION LAYER ---

        if executor is None:
            _prov_add_step(
                prov,
                log_step(
                    dataset_id=dataset_id,
                    step_id=step_id,
                    action_name=action_name,
                    approved=True,
                    status="skipped",
                    params=exec_params,
                    rationale=a.rationale,
                    before_schema=before_schema,
                    after_schema=None,
                    diff_summary=None,
                    warnings=None,
                    error="Action not implemented",
                ),
            )
            continue

        try:
            df_after, diff, warnings = executor(df_current, exec_params)
            after_schema = {c: str(df_after[c].dtype) for c in df_after.columns}

            _prov_add_step(
                prov,
                log_step(
                    dataset_id=dataset_id,
                    step_id=step_id,
                    action_name=action_name,
                    approved=True,
                    status="applied",
                    params=exec_params,
                    rationale=a.rationale,
                    before_schema=before_schema,
                    after_schema=after_schema,
                    diff_summary=diff,
                    warnings=warnings,
                    error=None,
                ),
            )
            df_current = df_after

        except Exception as e:
            _prov_add_step(
                prov,
                log_step(
                    dataset_id=dataset_id,
                    step_id=step_id,
                    action_name=action_name,
                    approved=True,
                    status="failed",
                    params=exec_params,
                    rationale=a.rationale,
                    before_schema=before_schema,
                    after_schema=None,
                    diff_summary=None,
                    warnings=None,
                    error=str(e),
                ),
            )
            continue

    return df_current, prov, plan

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate C1_manual.csv and C2/C3/C4 LLM plans for evaluation.")
    parser.add_argument("--root", type=str, default=".", help="Project root")
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=[
            "all",
            "phase_a", "phase_b",
            "adult", "diabetes", "student", "life_expectancy",
            "house_prices", "heart", "bank",
        ],
    )
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    parser.add_argument("--ollama-model", type=str, default="qwen2.5:3b-instruct")
    parser.add_argument("--debug-dir", type=str, default=None, help="Optional folder to dump raw/sanitized LLM outputs.")
    parser.add_argument("--llm-tag", type=str, default=None,
                        help="Short tag identifying this LLM run (e.g. 'qwen2.5_3b'). "
                             "Plan/provenance files are written with this suffix to allow side-by-side runs.")
    parser.add_argument("--condition", type=str, default="c2", choices=["c2", "c3", "c4"],
                        help="Which condition to generate: c2 (blind), c3 (context-aware), or c4 (expanded action space).")
    parser.add_argument("--user-context", type=str, default=None,
                        help="Path to a JSON file with user context for C3/C4. "
                             "Can be a single dict or a dict keyed by dataset_id.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    exports = root / "data" / "exports"
    debug_dir = Path(args.debug_dir).resolve() if args.debug_dir else None

    # Load user context if provided
    user_context_all: Optional[Dict[str, Any]] = None
    if args.user_context:
        ctx_path = Path(args.user_context)
        if ctx_path.exists():
            user_context_all = json.loads(ctx_path.read_text(encoding="utf-8"))
            print(f"[CTX] Loaded user context from {ctx_path}")
        else:
            print(f"[WARN] User context file not found: {ctx_path}")

    PHASE_A = ["adult", "diabetes", "student", "life_expectancy"]
    PHASE_B = ["house_prices", "heart", "bank"]
    if args.dataset == "all":
        to_run = PHASE_A
    elif args.dataset == "phase_a":
        to_run = PHASE_A
    elif args.dataset == "phase_b":
        to_run = PHASE_B
    else:
        to_run = [args.dataset]

    for ds in to_run:
        ds_dir = exports / ds
        c0_path = ds_dir / "c0_raw.csv"
        if not c0_path.exists():
            raise FileNotFoundError(f"Missing {c0_path}. Run data/scripts/prepare_datasets.py first.")

        print(f"\n[INFO] Preparing conditions for dataset: {ds}")
        df_c0 = pd.read_csv(c0_path)

        # Remove leakage columns
        leakage_cols = DATASET_SPECS[ds].get("leakage_cols", [])
        cols_to_drop = [col for col in leakage_cols if col in df_c0.columns]
        if cols_to_drop:
            df_c0 = df_c0.drop(columns=cols_to_drop)
            print(f"       [!] Removed Leakage columns: {cols_to_drop}")

        # C1
        df_c1, prov_c1 = apply_c1_pipeline(df_c0, dataset_id=ds)
        c1_path = ds_dir / "c1_manual.csv"
        df_c1.to_csv(c1_path, index=False)

        prov_dir = ds_dir / "provenance"
        _ensure_dir(prov_dir)
        (prov_dir / "c1_manual.json").write_text(json.dumps(_prov_to_dict(prov_c1), indent=2), encoding="utf-8")

        print(f"[OK] {ds}: saved {c1_path}")
        print(f"[OK] {ds}: saved {prov_dir / 'c1_manual.json'}")

        # C2/C3/C4
        target = DATASET_SPECS[ds]["target"]

        # Resolve per-dataset user context
        ds_user_context = None
        if args.condition in ("c3", "c4") and user_context_all is not None:
            if ds in user_context_all:
                ds_user_context = user_context_all[ds]
            else:
                ds_user_context = user_context_all

        if args.condition == "c4":
            cond_label = "C4_expanded"
        elif args.condition == "c3":
            cond_label = "C3_context"
        else:
            cond_label = "C2_llm"
        print(f"[INFO] {ds}: generating {cond_label} plan...")

        start_time = time.time()

        df_c2, prov_c2, plan = apply_c2_llm(
            df=df_c0,
            dataset_id=ds,
            target_column=target,
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model,
            debug_dir=debug_dir,
            user_context=ds_user_context,
            condition_mode=args.condition,
        )

        end_time = time.time()
        latency = round(end_time - start_time, 2)
        print(f"[METRIC] LLM Latency for {ds}: {latency} seconds")

        tag = args.llm_tag.strip() if args.llm_tag else None
        suffix = f"_{tag}" if tag else ""

        if args.condition == "c4":
            file_prefix = "c4_expanded"
        elif args.condition == "c3":
            file_prefix = "c3_context"
        else:
            file_prefix = "c2_llm"

        c_path = ds_dir / f"{file_prefix}{suffix}.csv"
        df_c2.to_csv(c_path, index=False)

        (prov_dir / f"{file_prefix}{suffix}.json").write_text(
            json.dumps(_prov_to_dict(prov_c2), indent=2), encoding="utf-8"
        )
        (prov_dir / f"{file_prefix}_plan{suffix}.json").write_text(
            json.dumps(plan.model_dump(), indent=2), encoding="utf-8"
        )
        (prov_dir / f"{file_prefix}_latency{suffix}.json").write_text(
            json.dumps({"latency_seconds": latency, "model": args.ollama_model,
                        "condition": args.condition,
                        "user_context": ds_user_context}, indent=2),
            encoding="utf-8",
        )

        prov_file = file_prefix + suffix + ".json"
        plan_file = file_prefix + "_plan" + suffix + ".json"
        print(f"[OK] {ds}: saved {c_path}")
        print(f"[OK] {ds}: saved {prov_dir / prov_file}")
        print(f"[OK] {ds}: saved {prov_dir / plan_file}")

    print("\n[DONE] Condition exports generated.")


if __name__ == "__main__":
    main()
