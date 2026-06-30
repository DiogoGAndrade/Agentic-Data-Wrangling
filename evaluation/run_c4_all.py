"""
Run all C4 (expanded action space) experiments.

This script iterates over every (dataset, ml_model) combination,
uses the matching C4 context JSON (with column_semantics), and calls
prepare_conditions followed by run_experiments for each.

Usage:
    python -m evaluation.run_c4_all --llm-tag qwen2.5_3b --ollama-model "qwen2.5:3b"

To run ALL 8 LLMs sequentially:
    python -m evaluation.run_c4_all --all-llms

To skip plan generation and only evaluate existing plans:
    python -m evaluation.run_c4_all --llm-tag qwen2.5_3b --ollama-model "qwen2.5:3b" --skip-prepare
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# LLMs to test (same as C3 runs)
ALL_LLMS: List[Tuple[str, str]] = [
    ("qwen2.5_3b",       "qwen2.5:3b"),
    ("llama3.2_3b",      "llama3.2:3b"),
    ("mistral_7b",       "mistral:7b"),
    ("qwen2.5_7b",       "qwen2.5:7b"),
    ("llama3.1_8b",      "llama3.1:8b"),
    ("gemma2_9b",        "gemma2:9b"),
    ("mistral_nemo_12b", "mistral-nemo:12b"),
    ("qwen2.5_14b",      "qwen2.5:14b"),
]

# Dataset × ML-model combinations (must match context file names in c4_contexts/)
PHASE_A_COMBOS: Dict[str, List[str]] = {
    "adult":            ["logreg", "rf", "knn", "gbm"],
    "diabetes":         ["logreg", "rf", "knn", "gbm"],
    "student":          ["logreg", "rf", "knn", "gbm"],
    "life_expectancy":  ["ridge", "rf", "knn", "gbm"],
}


def run_command(cmd: List[str], label: str) -> bool:
    """Run a subprocess and return True on success."""
    print(f"\n{'='*70}")
    print(f"[RUN] {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, cwd=str(Path(".").resolve()))
    if result.returncode != 0:
        print(f"[FAIL] {label} — exit code {result.returncode}")
        return False
    print(f"[OK] {label}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all C4 expanded-action-space experiments.")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--llm-tag", type=str, default=None,
                        help="Single LLM tag to run (e.g. 'qwen2.5_3b').")
    parser.add_argument("--ollama-model", type=str, default=None,
                        help="Ollama model name (e.g. 'qwen2.5:3b').")
    parser.add_argument("--all-llms", action="store_true",
                        help="Run all 8 LLMs sequentially.")
    parser.add_argument("--datasets", type=str, default="adult,diabetes,student,life_expectancy",
                        help="Comma-separated dataset list (Phase A training datasets).")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Skip plan generation (use existing C4 plans).")
    parser.add_argument("--debug-dir", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    contexts_dir = root / "evaluation" / "c4_contexts"
    requested_datasets = {d.strip() for d in args.datasets.split(",") if d.strip()}

    # Determine which LLMs to run
    if args.all_llms:
        llms_to_run = ALL_LLMS
    elif args.llm_tag and args.ollama_model:
        llms_to_run = [(args.llm_tag, args.ollama_model)]
    else:
        parser.error("Specify either --all-llms or both --llm-tag and --ollama-model.")
        return

    total_runs = 0
    successes = 0
    failures = []

    for llm_tag, ollama_model in llms_to_run:
        print(f"\n{'#'*70}")
        print(f"# LLM: {llm_tag} ({ollama_model})")
        print(f"{'#'*70}")

        for dataset_id, ml_models in PHASE_A_COMBOS.items():
            if dataset_id not in requested_datasets:
                continue

            for ml_model in ml_models:
                total_runs += 1
                ctx_file = contexts_dir / f"{dataset_id}_{ml_model}.json"

                if not ctx_file.exists():
                    print(f"[SKIP] Context file not found: {ctx_file.name}")
                    failures.append(f"{dataset_id}/{ml_model}/{llm_tag}: missing context")
                    continue

                label = f"{dataset_id} | {ml_model} | {llm_tag}"

                # Step 1: Generate C4 plan
                if not args.skip_prepare:
                    plan_tag = f"{llm_tag}_{ml_model}"
                    prepare_cmd = [
                        sys.executable, "-m", "evaluation.prepare_conditions",
                        "--root", str(root),
                        "--dataset", dataset_id,
                        "--ollama-url", args.ollama_url,
                        "--ollama-model", ollama_model,
                        "--llm-tag", plan_tag,
                        "--condition", "c4",
                        "--user-context", str(ctx_file),
                    ]
                    if args.debug_dir:
                        prepare_cmd.extend(["--debug-dir", args.debug_dir])

                    if not run_command(prepare_cmd, f"PREPARE C4 {label}"):
                        failures.append(f"{label}: prepare failed")
                        continue

                # Step 2: Evaluate the C4 plan
                plan_tag = f"{llm_tag}_{ml_model}"
                eval_cmd = [
                    sys.executable, "-m", "evaluation.run_experiments",
                    "--root", str(root),
                    "--llm-tag", plan_tag,
                    "--datasets", dataset_id,
                    "--condition", "C4",
                ]

                if run_command(eval_cmd, f"EVALUATE C4 {label}"):
                    successes += 1
                else:
                    failures.append(f"{label}: evaluate failed")

    # Summary
    print(f"\n{'='*70}")
    print(f"C4 EXPERIMENT RUN COMPLETE")
    print(f"{'='*70}")
    print(f"  Total combinations: {total_runs}")
    print(f"  Successes: {successes}")
    print(f"  Failures: {len(failures)}")
    if failures:
        print(f"\n  Failed runs:")
        for f in failures:
            print(f"    - {f}")

    # Reminder to consolidate
    print(f"\n[NEXT] Run: python -m evaluation.consolidate_results")
    print(f"       to merge C4 results into the MASTER table.")


if __name__ == "__main__":
    main()
