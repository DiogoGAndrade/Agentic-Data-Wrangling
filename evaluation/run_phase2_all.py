"""
Run Phase 2 (generalization) experiments on held-out test datasets.

This script:
1. Generates C2 plans for each LLM × test dataset (if not --skip-prepare)
2. Evaluates C2 on test datasets
3. Generates C3 plans for each LLM × ML model × test dataset
4. Evaluates C3 on test datasets
5. Consolidates all results into MASTER

Usage:
    # Run specific LLMs on all test datasets:
    python -m evaluation.run_phase2_all --llm-tag gemma2_9b --ollama-model "gemma2:9b"
    python -m evaluation.run_phase2_all --llm-tag qwen2.5_3b --ollama-model "qwen2.5:3b"

    # Run ALL LLMs (long!):
    python -m evaluation.run_phase2_all --all-llms

    # Skip plan generation (reuse existing plans):
    python -m evaluation.run_phase2_all --all-llms --skip-prepare
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


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

# Phase 2 test datasets
TEST_COMBOS: Dict[str, List[str]] = {
    "heart":        ["logreg", "rf", "knn", "gbm"],
    "bank":         ["logreg", "rf", "knn", "gbm"],
    "house_prices": ["ridge", "rf", "knn", "gbm"],
}


def run_command(cmd: List[str], label: str) -> bool:
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
    parser = argparse.ArgumentParser(description="Run Phase 2 generalization experiments.")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--llm-tag", type=str, default=None)
    parser.add_argument("--ollama-model", type=str, default=None)
    parser.add_argument("--all-llms", action="store_true")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Skip plan generation (use existing plans).")
    parser.add_argument("--skip-c2", action="store_true",
                        help="Skip C2 experiments (only run C3).")
    parser.add_argument("--skip-c3", action="store_true",
                        help="Skip C3 experiments (only run C2).")
    parser.add_argument("--debug-dir", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    contexts_dir = root / "evaluation" / "c3_contexts"
    test_datasets = ",".join(TEST_COMBOS.keys())

    if args.all_llms:
        llms_to_run = ALL_LLMS
    elif args.llm_tag and args.ollama_model:
        llms_to_run = [(args.llm_tag, args.ollama_model)]
    else:
        parser.error("Specify either --all-llms or both --llm-tag and --ollama-model.")
        return

    total = 0
    successes = 0
    failures = []

    for llm_tag, ollama_model in llms_to_run:
        print(f"\n{'#'*70}")
        print(f"# Phase 2 — LLM: {llm_tag} ({ollama_model})")
        print(f"{'#'*70}")

        # ---- C2: one plan per dataset, evaluate all ML models ----
        if not args.skip_c2:
            for dataset_id in TEST_COMBOS:
                total += 1
                label = f"C2 | {dataset_id} | {llm_tag}"

                if not args.skip_prepare:
                    prepare_cmd = [
                        sys.executable, "-m", "evaluation.prepare_conditions",
                        "--root", str(root),
                        "--dataset", dataset_id,
                        "--ollama-url", args.ollama_url,
                        "--ollama-model", ollama_model,
                        "--llm-tag", llm_tag,
                        "--condition", "c2",
                    ]
                    if args.debug_dir:
                        prepare_cmd.extend(["--debug-dir", args.debug_dir])
                    if not run_command(prepare_cmd, f"PREPARE {label}"):
                        failures.append(f"{label}: prepare failed")
                        continue

                eval_cmd = [
                    sys.executable, "-m", "evaluation.run_experiments",
                    "--root", str(root),
                    "--llm-tag", llm_tag,
                    "--datasets", dataset_id,
                    "--condition", "C2",
                ]
                if run_command(eval_cmd, f"EVALUATE {label}"):
                    successes += 1
                else:
                    failures.append(f"{label}: evaluate failed")

        # ---- C3: one plan per dataset×ML model, evaluate matching model ----
        if not args.skip_c3:
            for dataset_id, ml_models in TEST_COMBOS.items():
                for ml_model in ml_models:
                    total += 1
                    label = f"C3 | {dataset_id} | {ml_model} | {llm_tag}"
                    ctx_file = contexts_dir / f"{dataset_id}_{ml_model}.json"

                    if not ctx_file.exists():
                        print(f"[SKIP] Context file not found: {ctx_file.name}")
                        failures.append(f"{label}: missing context")
                        continue

                    if not args.skip_prepare:
                        plan_tag = f"{llm_tag}_{ml_model}"
                        prepare_cmd = [
                            sys.executable, "-m", "evaluation.prepare_conditions",
                            "--root", str(root),
                            "--dataset", dataset_id,
                            "--ollama-url", args.ollama_url,
                            "--ollama-model", ollama_model,
                            "--llm-tag", plan_tag,
                            "--condition", "c3",
                            "--user-context", str(ctx_file),
                        ]
                        if args.debug_dir:
                            prepare_cmd.extend(["--debug-dir", args.debug_dir])
                        if not run_command(prepare_cmd, f"PREPARE {label}"):
                            failures.append(f"{label}: prepare failed")
                            continue

                    plan_tag = f"{llm_tag}_{ml_model}"
                    eval_cmd = [
                        sys.executable, "-m", "evaluation.run_experiments",
                        "--root", str(root),
                        "--llm-tag", plan_tag,
                        "--datasets", dataset_id,
                        "--condition", "C3",
                    ]
                    if run_command(eval_cmd, f"EVALUATE {label}"):
                        successes += 1
                    else:
                        failures.append(f"{label}: evaluate failed")

    # Summary
    print(f"\n{'='*70}")
    print(f"PHASE 2 COMPLETE")
    print(f"{'='*70}")
    print(f"  Total: {total}  Success: {successes}  Fail: {len(failures)}")
    if failures:
        print(f"\n  Failures:")
        for f in failures:
            print(f"    - {f}")

    print(f"\n[NEXT] Run: python -m evaluation.consolidate_results")


if __name__ == "__main__":
    main()
