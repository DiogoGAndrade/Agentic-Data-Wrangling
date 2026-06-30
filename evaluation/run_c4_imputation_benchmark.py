"""
C4 Imputation Benchmark — Abordagem B (Online, full integration)
=================================================================
Reads the perturbed CSVs created by feature_level_test.py --mode online,
runs C4 (LLM plan via Ollama → PlanBasedCleaner) on each one, measures
MAE against ground truth, and produces the final Figure E with all 3 bars.

Run from project root:
    python evaluation/run_c4_imputation_benchmark.py --dataset adult --llm qwen2.5:3b
    python evaluation/run_c4_imputation_benchmark.py --dataset adult --llm qwen2.5:3b --all-models

Requires: Ollama running locally with the model pulled.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_error

# ── project root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.cleaning_pipeline import PlanBasedCleaner
from engine.config import RANDOM_STATE
from engine.profile_dataset import build_dataset_profile
from llm.ollama_client import OllamaClient
from evaluation.prepare_conditions import build_plan_prompt_c4

# ── paths ─────────────────────────────────────────────────────────────────────
EXPORTS = ROOT / "data" / "exports"
OUT_FIG = ROOT / "evaluation" / "outputs" / "figures"
OUT_CSV = ROOT / "evaluation" / "outputs"
OUT_FIG.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434"
RANDOM_STATE_NP = RANDOM_STATE

# ── dataset metadata ──────────────────────────────────────────────────────────
DATASET_SPECS = {
    "adult":           {"target": "income",          "task": "classification"},
    "diabetes":        {"target": "readmitted",      "task": "classification"},
    "student":         {"target": "final_result",    "task": "classification"},
    "life_expectancy": {"target": "life_expectancy", "task": "regression"},
    "house_prices":    {"target": "SalePrice",       "task": "regression"},
    "heart":           {"target": "target",          "task": "classification"},
    "bank":            {"target": "y",               "task": "classification"},
    # Phase C — high-missingness stress test
    "platform":        {"target": "purchased",       "task": "classification"},
    "support2_clf":    {"target": "hospdead",        "task": "classification"},
    "support2_reg":    {"target": "log_charges",     "task": "regression"},
}

# ── C4 user context per dataset (matches what prepare_conditions uses) ────────
USER_CONTEXT = {
    "adult": {
        "downstream_model": "RandomForest",
        "column_semantics": {
            "age": "numeric",
            "fnlwgt": "numeric",
            "education_num": "ordinal",
        },
        "redundant_features": [],
        "leakage_cols": [],
    },
    "life_expectancy": {
        "downstream_model": "RandomForest",
        "column_semantics": {},
        "redundant_features": [],
        "leakage_cols": [],
    },
    "diabetes": {
        "downstream_model": "RandomForest",
        "column_semantics": {},
        "redundant_features": [],
        "leakage_cols": ["encounter_id", "patient_nbr"],
    },
    "student": {
        "downstream_model": "RandomForest",
        "column_semantics": {},
        "redundant_features": [],
        "leakage_cols": ["id_student"],
    },
    "house_prices": {
        "downstream_model": "RandomForest",
        "column_semantics": {},
        "redundant_features": [],
        "leakage_cols": ["Id"],
    },
    "heart": {
        "downstream_model": "RandomForest",
        "column_semantics": {},
        "redundant_features": [],
        "leakage_cols": [],
    },
    "bank": {
        "downstream_model": "RandomForest",
        "column_semantics": {},
        "redundant_features": [],
        "leakage_cols": [],
    },
    # Phase C
    "platform": {
        "downstream_model": "RandomForest",
        "column_semantics": {
            "age": "numeric",
            "income": "numeric",
            "days_on_platform": "numeric",
            "gender": "categorical",
            "city": "categorical",
        },
        "redundant_features": [],
        "leakage_cols": [],
    },
    "support2_clf": {
        "downstream_model": "RandomForest",
        "column_semantics": {
            "age": "numeric",
            "meanbp": "numeric",
            "hrt": "numeric",
            "resp": "numeric",
            "temp": "numeric",
            "sps": "numeric",
            "aps": "numeric",
        },
        "redundant_features": [],
        "leakage_cols": [],
    },
    "support2_reg": {
        "downstream_model": "RandomForest",
        "column_semantics": {
            "age": "numeric",
            "meanbp": "numeric",
            "hrt": "numeric",
        },
        "redundant_features": [],
        "leakage_cols": [],
    },
}


def generate_c4_plan(df: pd.DataFrame, dataset: str, llm: str) -> dict:
    """
    Call Ollama with the C4 prompt to get a cleaning plan for this dataset.
    Returns the plan as a dict (compatible with PlanBasedCleaner).
    """
    spec = DATASET_SPECS[dataset]
    target = spec["target"]
    ctx = USER_CONTEXT.get(dataset, {})

    columns = list(df.columns)
    preview_rows = df.head(8).to_dict(orient="records")
    profile = build_dataset_profile(df, target_column=target)

    client = OllamaClient(base_url=OLLAMA_URL, model=llm)

    prompt = build_plan_prompt_c4(
        dataset_name=dataset,
        columns=columns,
        preview_rows=preview_rows,
        target_column=target,
        dataset_profile=profile,
        aggressive_filter=False,
        user_context=ctx,
    )

    print(f"    [LLM] Calling {llm} for {dataset} plan...", end=" ", flush=True)
    try:
        raw = client.generate(prompt)
        print("OK")
    except Exception as e:
        print(f"FAILED ({e})")
        return {"actions": []}

    # Parse JSON from response
    raw = raw.strip()
    # Find first { and last }
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        print("    [WARN] No JSON found in LLM response. Using empty plan.")
        return {"actions": []}

    try:
        plan_dict = json.loads(raw[start:end])
        n_actions = len(plan_dict.get("actions", []))
        print(f"    [Plan] {n_actions} actions parsed")
        return plan_dict
    except json.JSONDecodeError as e:
        print(f"    [WARN] JSON parse failed: {e}. Using empty plan.")
        return {"actions": []}


def apply_c4_and_measure(
    df_perturbed: pd.DataFrame,
    df_clean: pd.DataFrame,
    inject_idx: np.ndarray,
    col: str,
    dataset: str,
    llm: str,
    plan_cache: dict,
) -> float:
    """
    Apply C4 to df_perturbed, return MAE on the injected positions for col.
    plan_cache: dict to avoid re-calling LLM for the same dataset+llm.
    """
    cache_key = f"{dataset}_{llm}"
    if cache_key not in plan_cache:
        plan_cache[cache_key] = generate_c4_plan(df_perturbed, dataset, llm)

    plan = plan_cache[cache_key]
    target = DATASET_SPECS[dataset]["target"]

    # Split features / target for PlanBasedCleaner (it needs X without target)
    X = df_perturbed.drop(columns=[target], errors="ignore").copy()
    X_clean = df_clean.drop(columns=[target], errors="ignore").copy()

    try:
        cleaner = PlanBasedCleaner(plan=plan, target_column=target)
        cleaner.fit(X)
        X_recovered = cleaner.transform(X)

        if col not in X_recovered.columns:
            print(f"    [WARN] Column {col} not in C4 output. Returning NaN.")
            return np.nan

        gt_vals = X_clean[col].values
        c4_vals = X_recovered[col].values

        # Only score injected positions that C4 actually filled
        valid_mask = ~pd.isna(c4_vals[inject_idx])
        if valid_mask.sum() == 0:
            print(f"    [WARN] C4 left all injected values as NaN for {col}.")
            return np.nan

        mae = mean_absolute_error(
            gt_vals[inject_idx][valid_mask],
            c4_vals[inject_idx][valid_mask],
        )
        return float(mae)

    except Exception as e:
        print(f"    [ERROR] C4 failed on {col}: {e}")
        return np.nan


def run_benchmark(dataset: str, llm: str):
    """
    Full benchmark: reads existing benchmark_results.csv from perturbations/,
    fills in mae_c4, saves updated CSV, regenerates Figure E.
    """
    ds_dir = EXPORTS / dataset
    perturb_dir = ds_dir / "perturbations"
    bench_path = perturb_dir / "benchmark_results.csv"

    if not bench_path.exists():
        print(f"[ERROR] {bench_path} not found.")
        print("Run first: python evaluation/feature_level_test.py --mode online "
              f"--dataset {dataset} --llm {llm}")
        return

    df_res = pd.read_csv(bench_path)
    df_clean = pd.read_csv(ds_dir / "c0_raw.csv")

    # Numeric cols with 0 natural missings (same logic as feature_level_test.py)
    num_cols = df_clean.select_dtypes(include="number").columns
    clean_cols = [c for c in num_cols if df_clean[c].isna().sum() == 0][:3]

    plan_cache = {}
    missing_rates = [0.05, 0.10, 0.20]

    print(f"\n[C4 Integration] Dataset: {dataset} | LLM: {llm}")
    print(f"Columns: {clean_cols}")

    for col in clean_cols:
        gt_values = df_clean[col].values.copy()

        for rate in missing_rates:
            rng = np.random.default_rng(RANDOM_STATE_NP)
            n_inject = int(len(df_clean) * rate)
            inject_idx = rng.choice(len(df_clean), size=n_inject, replace=False)

            perturb_path = perturb_dir / f"perturbed_{col.replace(' ', '_')}_{int(rate * 100)}pct.csv"

            if not perturb_path.exists():
                print(f"  [WARN] {perturb_path.name} not found — skipping")
                continue

            df_perturbed = pd.read_csv(perturb_path)

            print(f"\n  col={col}, rate={int(rate*100)}%")
            mae_c4 = apply_c4_and_measure(
                df_perturbed, df_clean, inject_idx, col, dataset, llm, plan_cache
            )
            print(f"    mae_c4 = {mae_c4:.4f}" if not np.isnan(mae_c4) else "    mae_c4 = NaN")

            # Update the row in df_res
            mask = (df_res["column"] == col) & (df_res["missing_rate"] == rate)
            df_res.loc[mask, "mae_c4"] = mae_c4

    # Save updated benchmark CSV
    llm_tag = llm.replace(":", "_").replace(".", "_")
    out_bench = OUT_CSV / f"benchmark_c4_{dataset}_{llm_tag}.csv"
    df_res.to_csv(out_bench, index=False)
    print(f"\n[OK] Updated benchmark saved to {out_bench}")

    print("\n=== Final Benchmark (MAE) ===")
    print(df_res[["column", "missing_rate", "mae_mean_impute", "mae_mice", "mae_c4"]].to_string(index=False))

    _plot_final(df_res, dataset, llm)
    return df_res


def _plot_final(df_res: pd.DataFrame, dataset: str, llm: str):
    """Generate Figure E with all 3 bars filled (including C4)."""
    cols = df_res["column"].unique()
    fig, axes = plt.subplots(1, len(cols), figsize=(4.5 * len(cols), 4.8), sharey=False)
    if len(cols) == 1:
        axes = [axes]

    COLORS = {
        "Mean": "#9e9e9e",
        "MICE": "#f57c00",
        "C4":   "#1565C0",
    }

    bw = 0.25
    for ax, col in zip(axes, cols):
        sub = df_res[df_res["column"] == col].reset_index(drop=True)
        x = np.arange(len(sub))

        bars_mean = ax.bar(x - bw, sub["mae_mean_impute"], bw,
                           label="Mean impute", color=COLORS["Mean"], alpha=0.9)
        bars_mice = ax.bar(x,       sub["mae_mice"],        bw,
                           label="MICE",        color=COLORS["MICE"], alpha=0.9)
        bars_c4   = ax.bar(x + bw,  sub["mae_c4"],          bw,
                           label="C4 (LLM)",    color=COLORS["C4"], alpha=0.9)

        # Annotate C4 bars
        for i, (_, row) in enumerate(sub.iterrows()):
            if not np.isnan(row["mae_c4"]) and not np.isnan(row["mae_mean_impute"]):
                delta = row["mae_c4"] - row["mae_mean_impute"]
                color = "#1b5e20" if delta < 0 else "#b71c1c"
                sign = "−" if delta < 0 else "+"
                ax.text(i + bw, row["mae_c4"] + 0.01 * ax.get_ylim()[1],
                        f"{sign}{abs(delta):.2f}",
                        ha="center", va="bottom", fontsize=6.5,
                        color=color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(r*100)}% miss." for r in sub["missing_rate"]])
        ax.set_title(col[:28], fontsize=9)
        ax.set_ylabel("MAE (reconstruction error)")
        ax.legend(fontsize=7)

    llm_label = llm.replace(":", " ")
    fig.suptitle(
        f"Figure E — Missing Value Recovery Benchmark ({dataset})\n"
        f"C4 [{llm_label}] vs Mean Imputation vs MICE — lower is better",
        fontsize=9, y=1.02,
    )
    fig.tight_layout()

    llm_tag = llm.replace(":", "_").replace(".", "_")
    for ext in ("pdf", "png"):
        p = OUT_FIG / f"figureE_c4_benchmark_{dataset}_{llm_tag}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200 if ext == "png" else 150)
        print(f"[OK] {p}")
    plt.close(fig)


def run_all_models(dataset: str):
    """Run the benchmark across all available LLMs and produce a summary figure."""
    llms = [
        "qwen2.5:3b",
        "llama3.2:3b",
        "mistral:7b",
        "qwen2.5:7b",
        "llama3.1:8b",
        "gemma2:9b",
        "mistral-nemo:12b",
        "qwen2.5:14b",
    ]

    all_results = []
    for llm in llms:
        print(f"\n{'='*60}")
        print(f"LLM: {llm}")
        print("="*60)
        df = run_benchmark(dataset, llm)
        if df is not None:
            df["llm"] = llm
            all_results.append(df)

    if not all_results:
        print("[ERROR] No results collected.")
        return

    df_all = pd.concat(all_results, ignore_index=True)
    out = OUT_CSV / f"benchmark_c4_all_llms_{dataset}.csv"
    df_all.to_csv(out, index=False)
    print(f"\n[OK] All-LLM benchmark saved to {out}")

    # Summary figure: mean MAE across columns per LLM
    _plot_scaling_benchmark(df_all, dataset)


def _plot_scaling_benchmark(df_all: pd.DataFrame, dataset: str):
    """Figure F — C4 MAE vs LLM size (scaling law for imputation quality)."""
    MODEL_SIZES = {
        "qwen2.5:3b":       3,
        "llama3.2:3b":      3,
        "mistral:7b":       7,
        "qwen2.5:7b":       7,
        "llama3.1:8b":      8,
        "gemma2:9b":        9,
        "mistral-nemo:12b": 12,
        "qwen2.5:14b":      14,
    }

    summary = (
        df_all.groupby("llm")["mae_c4"]
        .mean()
        .reset_index()
        .rename(columns={"mae_c4": "mean_mae"})
    )
    summary["size_b"] = summary["llm"].map(MODEL_SIZES)
    summary = summary.dropna().sort_values("size_b")

    # Also add Mean/MICE baselines (same for all LLMs)
    baseline_mean = df_all["mae_mean_impute"].mean()
    baseline_mice = df_all["mae_mice"].mean()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(summary["size_b"], summary["mean_mae"], "o-",
            color="#1565C0", lw=2, ms=7, label="C4 (LLM)")
    ax.axhline(baseline_mean, color="#9e9e9e", ls="--", lw=1.2, label="Mean impute")
    ax.axhline(baseline_mice, color="#f57c00", ls="--", lw=1.2, label="MICE")

    for _, row in summary.iterrows():
        ax.annotate(row["llm"].split(":")[0],
                    (row["size_b"], row["mean_mae"]),
                    textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7)

    ax.set_xlabel("Model size (B parameters)")
    ax.set_ylabel("Mean MAE (across columns & injection rates)")
    ax.set_title(
        f"Figure F — C4 Imputation Quality vs LLM Size ({dataset})\n"
        "Scaling law: does more parameters → better imputation?",
        fontsize=9,
    )
    ax.legend()
    fig.tight_layout()

    for ext in ("pdf", "png"):
        p = OUT_FIG / f"figureF_scaling_imputation_{dataset}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200 if ext == "png" else 150)
        print(f"[OK] {p}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run C4 integration benchmark for missing value recovery."
    )
    parser.add_argument("--dataset", default="platform",
                        choices=list(DATASET_SPECS.keys()),
                        help="Dataset to benchmark")
    parser.add_argument("--llm", default="qwen2.5:3b",
                        help="Ollama model tag (e.g. qwen2.5:3b, mistral:7b)")
    parser.add_argument("--all-models", action="store_true",
                        help="Run all 8 LLMs and produce scaling-law figure")
    args = parser.parse_args()

    if args.all_models:
        run_all_models(args.dataset)
    else:
        run_benchmark(args.dataset, args.llm)
