"""
Plan-quality metrics for thesis Section 5.4 / 3.5.4.

For each (dataset, llm_model) pair, reads:
  - data/exports/<ds>/provenance/c2_llm_plan.json   (the LLM-proposed plan)
  - data/exports/<ds>/provenance/c2_llm.json        (the execution provenance)

Computes:
  - n_actions_proposed
  - n_actions_applied
  - n_actions_skipped
  - n_actions_failed
  - rejection_rate    = (skipped + failed) / proposed
  - structurally_valid: did the plan parse without falling back to default?
  - latency_seconds   (if recorded in provenance)
  - distinct action types proposed

Outputs a CSV at evaluation/outputs/plan_quality_<llm_model>.csv
and a consolidated MASTER_PLAN_QUALITY.csv.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

DATASETS = ["adult", "diabetes", "student", "life_expectancy", "house_prices", "heart", "bank"]


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def compute_plan_quality(
    ds: str,
    llm_model: str,
    plan_path: Path,
    prov_path: Path,
) -> Dict[str, object]:
    plan = _load_json(plan_path)
    prov = _load_json(prov_path)

    row: Dict[str, object] = {
        "dataset": ds,
        "llm_model": llm_model,
        "n_actions_proposed": 0,
        "n_actions_applied": 0,
        "n_actions_skipped": 0,
        "n_actions_failed": 0,
        "rejection_rate": None,
        "structurally_valid": False,
        "fallback_triggered": False,
        "distinct_action_types": "",
        "latency_seconds": None,
    }

    if plan is None or prov is None:
        return row

    # ---- Plan side ----
    actions = plan.get("actions", []) or []
    row["n_actions_proposed"] = len(actions)
    types_proposed = [a.get("action") for a in actions if isinstance(a, dict)]
    counter = Counter(types_proposed)
    row["distinct_action_types"] = ";".join(f"{k}:{v}" for k, v in sorted(counter.items()))

    # If the plan only contains the safety fallback (single fix_column_names with that exact rationale),
    # we treat the LLM call as having failed structurally.
    fallback_signature = (
        len(actions) == 1
        and actions[0].get("action") == "fix_column_names"
        and "fallback" in (actions[0].get("rationale", "").lower())
    )
    row["fallback_triggered"] = fallback_signature
    row["structurally_valid"] = not fallback_signature

    # ---- Provenance side ----
    steps = prov.get("steps", []) or []

    applied = sum(1 for s in steps if s.get("status") == "applied" and s.get("action_name") != "llm_plan_parse")
    skipped = sum(1 for s in steps if s.get("status") == "skipped" and s.get("action_name") != "llm_plan_parse")
    failed = sum(1 for s in steps if s.get("status") == "failed" and s.get("action_name") != "llm_plan_parse")

    row["n_actions_applied"] = applied
    row["n_actions_skipped"] = skipped
    row["n_actions_failed"] = failed

    proposed = max(row["n_actions_proposed"], applied + skipped + failed)
    if proposed > 0:
        row["rejection_rate"] = round((skipped + failed) / proposed, 4)

    # Optional latency (if recorded as a step or sidecar).
    # Try multiple naming conventions for the latency file.
    latency_candidates = [
        plan_path.parent / f"c2_llm_latency_{llm_model}.json",
        plan_path.parent / f"c4_expanded_latency_{llm_model}.json",
        plan_path.parent / "c2_llm_latency.json",
    ]
    for latency_sidecar in latency_candidates:
        lat = _load_json(latency_sidecar)
        if isinstance(lat, dict) and "latency_seconds" in lat:
            row["latency_seconds"] = lat["latency_seconds"]
            break

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute plan-quality metrics per (dataset, LLM model).")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument(
        "--llm-model",
        type=str,
        required=True,
        help="Tag identifying the LLM run, e.g. 'qwen2.5_3b' or 'mistral_nemo'. "
             "Provenance files for that run must already be in data/exports/<ds>/provenance/.",
    )
    parser.add_argument(
        "--master-out",
        type=str,
        default=None,
        help="If set, append rows to this consolidated CSV (creates if absent).",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="c2",
        choices=["c2", "c3", "c4"],
        help="Which condition's plan files to read (c2, c3, c4). Default: c2.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of datasets. Default: all in DATASETS.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rows: List[Dict[str, object]] = []

    ds_list = [d.strip() for d in args.datasets.split(",")] if args.datasets else DATASETS

    for ds in ds_list:
        prov_dir = root / "data" / "exports" / ds / "provenance"

        # Resolve plan/provenance paths based on condition
        tag = args.llm_model
        if args.condition == "c4":
            plan_path = prov_dir / f"c4_expanded_plan_{tag}.json"
            prov_path = prov_dir / f"c4_expanded_{tag}.json"
        elif args.condition == "c3":
            plan_path = prov_dir / f"c3_context_plan_{tag}.json"
            prov_path = prov_dir / f"c3_context_{tag}.json"
        else:
            plan_path = prov_dir / "c2_llm_plan.json"
            prov_path = prov_dir / "c2_llm.json"

        if not plan_path.exists():
            print(f"[SKIP] {ds}: plan not found at {plan_path.name}")
            continue

        rows.append(compute_plan_quality(ds, args.llm_model, plan_path, prov_path))

    df = pd.DataFrame(rows)
    out_dir = root / "evaluation" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"plan_quality_{args.llm_model}.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] {out_path}")
    print(df.to_string(index=False))

    # Optional consolidated master file
    master_path = Path(args.master_out).resolve() if args.master_out else (out_dir / "MASTER_PLAN_QUALITY.csv")
    if master_path.exists():
        prev = pd.read_csv(master_path)
        merged = pd.concat([prev, df], ignore_index=True).drop_duplicates(
            subset=["dataset", "llm_model"], keep="last"
        )
    else:
        merged = df
    merged.to_csv(master_path, index=False)
    print(f"[OK] Master plan-quality table: {master_path}")


if __name__ == "__main__":
    main()
