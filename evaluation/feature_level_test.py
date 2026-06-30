"""
Feature-Level Ablation Test — Agentic Data Wrangling
=====================================================
Tests C4's ability to handle specific data quality issues in isolation.

TWO MODES:
  1. OFFLINE (no Ollama needed): Compares C4-exported data vs C0 to measure
     what the system actually did to each quality type. Produces Figure D.

  2. ONLINE (requires Ollama): Injects controlled perturbations into a clean
     dataset, applies C4, measures recovery. Produces Figure E.

Run offline mode (no Ollama):
    python evaluation/feature_level_test.py --mode offline

Run online mode (requires Ollama running):
    python evaluation/feature_level_test.py --mode online --dataset life_expectancy --llm qwen2.5:3b

Output: evaluation/outputs/figures/
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "figures")
os.makedirs(OUT, exist_ok=True)

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "exports")
RANDOM_STATE = 369

PLT_STYLE = {
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
}
plt.rcParams.update(PLT_STYLE)


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 — OFFLINE: What did C4 actually do to each column?
# ─────────────────────────────────────────────────────────────────────────────
def offline_analysis():
    """
    For each dataset with both c0_raw.csv and a c4_expanded_*.csv,
    measure:
    - Missing values filled (per column)
    - Columns added/removed
    - Outlier reduction (IQR-based)
    - Mean/std change per numeric column
    """
    datasets = {
        "life_expectancy": ("c0_raw.csv", "c4_expanded_qwen2.5_14b_knn.csv"),
        "adult":           ("c0_raw.csv", "c4_expanded_qwen2.5_14b_rf.csv"),
        "student":         ("c0_raw.csv", "c4_expanded_qwen2.5_14b_rf.csv"),
        "house_prices":    ("c0_raw.csv", "c4_expanded_qwen2.5_14b_rf.csv"),
    }

    results = []
    for ds, (c0_f, c4_f) in datasets.items():
        c0_path = os.path.join(BASE, ds, c0_f)
        c4_path = os.path.join(BASE, ds, c4_f)
        if not os.path.exists(c0_path) or not os.path.exists(c4_path):
            print(f"  [SKIP] {ds}: files not found")
            continue

        c0 = pd.read_csv(c0_path)
        c4 = pd.read_csv(c4_path)

        total_missing_c0 = c0.isna().sum().sum()
        total_missing_c4 = sum(c4[col].isna().sum() for col in c0.columns if col in c4.columns)
        cols_added   = len([c for c in c4.columns if c not in c0.columns])
        cols_removed = len([c for c in c0.columns if c not in c4.columns])

        # Outlier count: values beyond 3 IQR from median per numeric column
        def count_outliers(df):
            n = 0
            for col in df.select_dtypes(include='number').columns:
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    n += ((df[col] < q1 - 3*iqr) | (df[col] > q3 + 3*iqr)).sum()
            return n

        out_c0 = count_outliers(c0)
        out_c4 = count_outliers(c4[[c for c in c0.columns if c in c4.columns]])

        results.append({
            "dataset": ds,
            "missing_c0": total_missing_c0,
            "missing_c4": total_missing_c4,
            "missing_filled": total_missing_c0 - total_missing_c4,
            "missing_fill_rate": (total_missing_c0 - total_missing_c4) / max(total_missing_c0, 1) * 100,
            "outliers_c0": out_c0,
            "outliers_c4": out_c4,
            "outlier_reduction": out_c0 - out_c4,
            "cols_added": cols_added,
            "cols_removed": cols_removed,
        })
        print(f"  [OK] {ds}: {total_missing_c0} missing → {total_missing_c4} | {out_c0} outliers → {out_c4}")

    if not results:
        print("No data found. Check paths.")
        return

    df_r = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(df_r[["dataset","missing_filled","missing_fill_rate","outlier_reduction","cols_added","cols_removed"]].to_string(index=False))

    # ── Figure D ──────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ds_labels = df_r["dataset"].tolist()
    x = np.arange(len(ds_labels))
    bw = 0.35

    # Missing values
    ax1.bar(x - bw/2, df_r["missing_c0"], bw, label="C0 (before)", color="#9e9e9e", alpha=0.9)
    ax1.bar(x + bw/2, df_r["missing_c4"], bw, label="C4 (after)",  color="#1565C0", alpha=0.9)
    for i, row in df_r.iterrows():
        ax1.text(i + bw/2, row["missing_c4"] + 0.5,
                 f"{row['missing_fill_rate']:.0f}% filled",
                 ha="center", va="bottom", fontsize=6.5, color="#1565C0")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ds_labels, rotation=15, ha="right")
    ax1.set_ylabel("Total missing values")
    ax1.set_title("Missing Value Recovery by C4\n(after applying LLM-guided imputation plan)")
    ax1.legend()

    # Outliers
    ax2.bar(x - bw/2, df_r["outliers_c0"], bw, label="C0 (before)", color="#9e9e9e", alpha=0.9)
    ax2.bar(x + bw/2, df_r["outliers_c4"], bw, label="C4 (after)",  color="#1565C0", alpha=0.9)
    for i, row in df_r.iterrows():
        red = row["outlier_reduction"]
        if red > 0:
            ax2.text(i + bw/2, row["outliers_c4"] + 0.5,
                     f"-{red}", ha="center", va="bottom", fontsize=6.5, color="#1b5e20")
    ax2.set_xticks(x)
    ax2.set_xticklabels(ds_labels, rotation=15, ha="right")
    ax2.set_ylabel("Outlier count (3xIQR criterion)")
    ax2.set_title("Outlier Reduction by C4\n(values beyond 3×IQR from median)")
    ax2.legend()

    fig.suptitle("Figure D — Feature-Level Analysis: C4's direct effect on data quality\n"
                 "(measured on exported C4 vs C0 datasets, no ML evaluation)", fontsize=9, y=1.01)
    fig.tight_layout()
    for ext in ("pdf","png"):
        p = os.path.join(OUT, f"figureD_feature_level_offline.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=200 if ext=="png" else 150)
    plt.close(fig)
    print(f"\n[OK] Figure D saved to {OUT}")
    return df_r


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — ONLINE: Controlled injection + C4 recovery
# ─────────────────────────────────────────────────────────────────────────────
def online_analysis(dataset="life_expectancy", llm="qwen2.5:3b"):
    """
    1. Take c0_raw.csv (the real raw data).
    2. Pick 3 numeric columns with fewest natural missings.
    3. Inject MCAR missings at 5%, 10%, 20% → save perturbed versions.
    4. For each perturbed version, apply C4 imputation via Ollama.
    5. Compare recovered values to ground truth.
    6. Benchmark against SimpleImputer (mean) and MICE.

    NOTE: Step 4 requires Ollama running locally.
    Steps 1-3 and 5-6 can be run standalone.
    """
    c0_path = os.path.join(BASE, dataset, "c0_raw.csv")
    if not os.path.exists(c0_path):
        print(f"[ERROR] {c0_path} not found")
        return

    df_clean = pd.read_csv(c0_path)

    # Pick numeric columns with 0 natural missings
    num_cols = df_clean.select_dtypes(include='number').columns
    clean_cols = [c for c in num_cols if df_clean[c].isna().sum() == 0][:3]
    print(f"Selected columns for injection: {clean_cols}")

    missing_rates = [0.05, 0.10, 0.20]
    results = []

    for col in clean_cols:
        gt_values = df_clean[col].values.copy()

        for rate in missing_rates:
            rng = np.random.default_rng(RANDOM_STATE)
            n_inject = int(len(df_clean) * rate)
            inject_idx = rng.choice(len(df_clean), size=n_inject, replace=False)

            df_perturbed = df_clean.copy()
            df_perturbed.loc[inject_idx, col] = np.nan

            # Benchmark 1: Mean imputation
            mean_val = df_perturbed[col].mean()
            mean_imputed = df_perturbed[col].fillna(mean_val).values
            mae_mean = mean_absolute_error(gt_values[inject_idx], mean_imputed[inject_idx])

            # Benchmark 2: MICE (IterativeImputer)
            other_num = [c for c in num_cols if c != col and df_perturbed[c].isna().sum() == 0][:5]
            if other_num:
                mice_df = df_perturbed[[col] + other_num].copy()
                imp = IterativeImputer(max_iter=10, random_state=RANDOM_STATE)
                mice_imputed = imp.fit_transform(mice_df)[:, 0]
                mae_mice = mean_absolute_error(gt_values[inject_idx], mice_imputed[inject_idx])
            else:
                mae_mice = np.nan

            results.append({
                "column": col, "missing_rate": rate,
                "mae_mean_impute": mae_mean,
                "mae_mice": mae_mice,
                "mae_c4": np.nan,  # Fill after running C4 with Ollama
                "n_injected": n_inject,
            })

            # Save perturbed file for C4 to process
            perturb_dir = os.path.join(os.path.dirname(c0_path), "perturbations")
            os.makedirs(perturb_dir, exist_ok=True)
            out_path = os.path.join(perturb_dir, f"perturbed_{col.replace(' ','_')}_{int(rate*100)}pct.csv")
            df_perturbed.to_csv(out_path, index=False)

    df_res = pd.DataFrame(results)
    print("\n=== Imputation Benchmark (MAE) ===")
    print(df_res[["column","missing_rate","mae_mean_impute","mae_mice","mae_c4"]].to_string(index=False))

    # Save benchmark results
    bench_path = os.path.join(os.path.dirname(c0_path), "perturbations", "benchmark_results.csv")
    df_res.to_csv(bench_path, index=False)
    print(f"\nPerturbed files saved to: {os.path.join(os.path.dirname(c0_path), 'perturbations')}")
    print("Next step: run C4 on each perturbed file, then fill 'mae_c4' column and re-run plot_online_results()")

    _plot_benchmark(df_res, dataset)
    return df_res


def _plot_benchmark(df_res, dataset):
    """Plot imputation benchmark (C4 column will be NaN until Ollama is run)."""
    cols = df_res["column"].unique()
    fig, axes = plt.subplots(1, len(cols), figsize=(4.5 * len(cols), 4.5), sharey=False)
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        sub = df_res[df_res["column"] == col]
        x = np.arange(len(sub))
        bw = 0.25
        ax.bar(x - bw, sub["mae_mean_impute"], bw, label="Mean impute", color="#9e9e9e")
        ax.bar(x,       sub["mae_mice"],        bw, label="MICE",        color="#f57c00")
        ax.bar(x + bw,  sub["mae_c4"],          bw, label="C4 (LLM)",    color="#1565C0",
               alpha=0.5, hatch="//", edgecolor="#1565C0")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(r*100)}% missing" for r in sub["missing_rate"]])
        ax.set_title(col[:25])
        ax.set_ylabel("MAE (reconstruction)")
        ax.legend(fontsize=7)
        ax.text(0.5, 0.92, "C4 bar = placeholder\n(run Ollama to fill)",
                transform=ax.transAxes, ha="center", fontsize=6.5,
                color="#b71c1c", style="italic")

    fig.suptitle(f"Figure E — Missing Value Recovery Benchmark ({dataset})\n"
                 "C4 vs Mean Imputation vs MICE across injection rates",
                 fontsize=9, y=1.01)
    fig.tight_layout()
    for ext in ("pdf","png"):
        p = os.path.join(OUT, f"figureE_imputation_benchmark_{dataset}.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=200 if ext=="png" else 150)
    plt.close(fig)
    print(f"[OK] Figure E (benchmark) saved to {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["offline","online"], default="offline")
    parser.add_argument("--dataset", default="life_expectancy")
    parser.add_argument("--llm", default="qwen2.5:3b")
    args = parser.parse_args()

    if args.mode == "offline":
        print("Running offline feature-level analysis...")
        offline_analysis()
    else:
        print(f"Running online injection test on {args.dataset} with {args.llm}...")
        online_analysis(args.dataset, args.llm)
