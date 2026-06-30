"""
Thesis Figure Generator — Agentic Data Wrangling
Generates Figures 1 and 2 for thesis Chapter 5.

Run from project root:
    python evaluation/generate_thesis_figures.py

Output: evaluation/outputs/figures/
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

COLORS = {
    "C0":      "#9e9e9e",
    "C4":      "#1565C0",
    "FLAML":   "#2e7d32",
    "Optuna":  "#558b2f",
    "ChatGPT": "#e65100",
    "Claude":  "#b71c1c",
}

def load_best(master_csv, prefix):
    df = pd.read_csv(master_csv)
    sub = df[df["condition"].fillna("").str.startswith(prefix)].copy()
    results = {}
    for ds, grp in sub.groupby("dataset"):
        task = grp["task_type"].iloc[0]
        if task == "classification":
            idx  = grp["f1_macro"].idxmax()
            best = grp.loc[idx, "f1_macro"]
            std  = grp.loc[idx, "f1_macro_std"]
        else:
            idx  = grp["r2"].idxmax()
            best = grp.loc[idx, "r2"]
            std  = grp.loc[idx, "r2_std"]
        results[ds] = {"score": float(best), "std": float(std) if std == std else 0.0, "task": task}
    return results


def fig1_scores_per_dataset():
    base       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    master     = os.path.join(base, "MASTER_RESULTS_TABLE.csv")
    flaml_csv  = os.path.join(base, "results_flaml.csv")
    optuna_csv = os.path.join(base, "results_optuna.csv")
    cloud_csv  = os.path.join(base, "results_cloud_llm_all.csv")

    c4 = load_best(master, "C4_")
    c0 = load_best(master, "C0_raw")

    df_flaml  = pd.read_csv(flaml_csv)
    df_optuna = pd.read_csv(optuna_csv)
    df_cloud  = pd.read_csv(cloud_csv)

    flaml = {}
    for _, row in df_flaml.iterrows():
        ds    = row["dataset"]
        score = row["f1_macro"] if row["task_type"] == "classification" else row["r2"]
        std   = row["f1_macro_std"] if row["task_type"] == "classification" else row["r2_std"]
        if pd.notna(score):
            flaml[ds] = {"score": float(score), "std": float(std) if pd.notna(std) else 0.0}

    optuna = {}
    for _, row in df_optuna.iterrows():
        ds    = row["dataset"]
        score = row["f1_macro"] if row["task_type"] == "classification" else row["r2"]
        std   = row["f1_macro_std"] if row["task_type"] == "classification" else row["r2_std"]
        if pd.notna(score):
            optuna[ds] = {"score": float(score), "std": float(std) if pd.notna(std) else 0.0}

    chatgpt_scores = {}
    claude_scores  = {}
    for llm, grp in df_cloud.groupby("llm_tag"):
        for ds, ds_grp in grp.groupby("dataset"):
            idx  = ds_grp["f1_macro"].idxmax()
            best = ds_grp.loc[idx, "f1_macro"]
            std  = ds_grp.loc[idx, "f1_macro_std"]
            entry = {"score": float(best), "std": float(std) if pd.notna(std) else 0.0}
            if llm == "chatgpt":
                chatgpt_scores[ds] = entry
            elif llm == "claude":
                claude_scores[ds] = entry

    clf_order = ["adult", "bank", "diabetes", "heart", "student"]
    reg_order = ["life_expectancy", "house_prices"]
    datasets  = [d for d in clf_order + reg_order if d in c4]

    systems = ["C0", "C4", "FLAML", "Optuna", "ChatGPT", "Claude"]
    lookup  = {
        "C0": c0, "C4": c4, "FLAML": flaml,
        "Optuna": optuna, "ChatGPT": chatgpt_scores, "Claude": claude_scores,
    }

    data = {s: [] for s in systems}
    errs = {s: [] for s in systems}
    for ds in datasets:
        for s in systems:
            entry = lookup[s].get(ds, {})
            data[s].append(entry.get("score", np.nan))
            errs[s].append(entry.get("std", 0.0))

    n_ds  = len(datasets)
    n_sys = len(systems)
    bar_w = 0.13
    offsets = np.linspace(-(n_sys - 1) / 2 * bar_w, (n_sys - 1) / 2 * bar_w, n_sys)
    x = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, s in enumerate(systems):
        vals  = np.array(data[s], dtype=float)
        errs_ = np.array(errs[s], dtype=float)
        ax.bar(
            x + offsets[i], vals, bar_w,
            label=s, color=COLORS[s],
            yerr=errs_,
            error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": "#333333"},
            zorder=3, alpha=0.9,
        )

    clf_count = sum(1 for d in datasets if d in clf_order)
    if clf_count < n_ds:
        ax.axvline(clf_count - 0.5, color="#555555", lw=1, ls=":", alpha=0.7)

    label_map = {
        "adult": "Adult", "bank": "Bank", "diabetes": "Diabetes",
        "heart": "Heart", "student": "Student",
        "life_expectancy": "Life Exp.", "house_prices": "House Prices",
    }
    ax.set_xticks(x)
    ax.set_xticklabels([label_map.get(d, d) for d in datasets])
    ax.set_ylabel("Score (f1_macro / R2)")
    ax.set_title(
        "Figure 1 - Performance Comparison Across Systems and Datasets\n"
        "(C4 best across all LLM tags; error bars = +/-1 std across CV folds)"
    )
    ax.legend(loc="lower right", ncol=3, framealpha=0.9)

    all_vals = [v for s in systems for v in data[s] if not np.isnan(v)]
    ymin = max(0, min(all_vals) - 0.04)
    ax.set_ylim(ymin, min(1.0, max(all_vals) + 0.05))

    fig.tight_layout()
    pdf = os.path.join(OUT_DIR, "figure1_scores_per_dataset.pdf")
    png = os.path.join(OUT_DIR, "figure1_scores_per_dataset.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("[OK] Figure 1 PDF: " + pdf)
    print("[OK] Figure 1 PNG: " + png)


def fig2_wtl_stacked():
    systems = [
        ("C4 (Ours)",       3, 25, 0, True,  7),
        ("FLAML",           0,  7, 0, False, 7),
        ("mljar",           0,  7, 0, False, 7),
        ("Optuna+LGB",      0,  7, 0, False, 7),
        ("BayesSearch+RF",  0,  7, 0, False, 7),
        ("ChatGPT",         0,  4, 0, False, 4),
        ("Gemini",          0,  4, 0, False, 4),
        ("Claude z.s.",     0,  3, 1, False, 4),
        ("Copilot z.s.",    0,  3, 1, False, 4),
    ]

    labels = [s[0] for s in systems]
    wins   = [s[1] for s in systems]
    ties   = [s[2] for s in systems]
    losses = [s[3] for s in systems]
    guaran = [s[4] for s in systems]

    x     = np.arange(len(labels))
    bar_w = 0.55

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.bar(x, wins,  bar_w, label="WINs",   color="#1b5e20", zorder=3)
    ax.bar(x, ties,  bar_w, bottom=wins,    label="TIEs",   color="#b0bec5", zorder=3)
    ax.bar(x, losses, bar_w,
           bottom=[w + t for w, t in zip(wins, ties)],
           label="LOSSes", color="#b71c1c", zorder=3)

    for i, g in enumerate(guaran):
        total = wins[i] + ties[i] + losses[i]
        if g:
            ax.text(i, total + 0.3, "arch. guaranteed",
                    ha="center", va="bottom", fontsize=6.5,
                    color="#1b5e20", fontstyle="italic")

    for i in range(len(labels)):
        if wins[i]:
            ax.text(i, wins[i] / 2, str(wins[i]),
                    ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        if ties[i]:
            ax.text(i, wins[i] + ties[i] / 2, str(ties[i]),
                    ha="center", va="center", fontsize=8, color="#333333")
        if losses[i]:
            ax.text(i, wins[i] + ties[i] + losses[i] / 2, str(losses[i]),
                    ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Number of dataset comparisons")
    ax.set_title(
        "Figure 2 - Win/Tie/Loss Scorecard per System vs C0 Raw Baseline\n"
        "(threshold = variance-adjusted per dataset; n=4 for cloud LLMs, n=7 for others)"
    )
    ax.legend(loc="upper right", framealpha=0.9)

    totals = [wins[i] + ties[i] + losses[i] for i in range(len(labels))]
    ax.set_ylim(0, max(totals) + 2.5)

    ax.axvline(4.5, color="#555555", lw=0.8, ls=":", alpha=0.6)
    ax.text(4.6, 1, "cloud LLMs", fontsize=7, color="#555555", va="bottom")

    xlabels = ax.get_xticklabels()
    xlabels[0].set_fontweight("bold")
    xlabels[0].set_color("#1565C0")

    fig.tight_layout()
    pdf = os.path.join(OUT_DIR, "figure2_wtl_stacked.pdf")
    png = os.path.join(OUT_DIR, "figure2_wtl_stacked.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("[OK] Figure 2 PDF: " + pdf)
    print("[OK] Figure 2 PNG: " + png)


if __name__ == "__main__":
    print("Generating thesis figures...")
    fig1_scores_per_dataset()
    fig2_wtl_stacked()
    print("Done. Figures saved to: " + OUT_DIR)
