"""
Final thesis figures — Agentic Data Wrangling
Generates Figures A, B, C for Chapter 5.

Figure A: C0 vs C4 per dataset × ML model (4 subplots by model), with delta annotations.
Figure B: C4 vs all external comparators (4 common datasets, all 4 cloud LLMs visible).
Figure C: Delta heatmap — C4 improvement over C0 per dataset × ML model.

Run: python evaluation/generate_figures_final.py
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "figures")
os.makedirs(OUT, exist_ok=True)

BASE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
FLAML = os.path.join(BASE, "results_flaml.csv")
OPT   = os.path.join(BASE, "results_optuna.csv")
MASTER = os.path.join(BASE, "MASTER_RESULTS_TABLE.csv")

PLT_STYLE = {
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
}
plt.rcParams.update(PLT_STYLE)

C = {
    "C0":       "#9e9e9e",
    "C4":       "#1565C0",
    "FLAML":    "#2e7d32",
    "Optuna":   "#66bb6a",
    "ChatGPT":  "#FF8C00",
    "Gemini":   "#9C27B0",
    "Claude":   "#b71c1c",
    "Copilot":  "#795548",
}

DS_LABEL = {
    "adult": "Adult", "bank": "Bank", "diabetes": "Diabetes",
    "heart": "Heart", "student": "Student",
    "life_expectancy": "Life Exp.", "house_prices": "House Prices",
}
MODEL_LABEL = {"logreg": "LogReg", "rf": "RF", "knn": "KNN",
               "gbm": "GBM", "ridge": "Ridge"}

def load():
    df = pd.read_csv(MASTER)
    df["condition"] = df["condition"].fillna("")
    return df

def get_c4_best(df, ds, model):
    c4 = df[(df["dataset"]==ds) & (df["model"]==model) &
            df["condition"].str.startswith("C4_")]
    if c4.empty: return np.nan, 0.0
    task = c4["task_type"].iloc[0]
    m = "f1_macro" if task=="classification" else "r2"
    idx = c4[m].idxmax()
    return float(c4.loc[idx, m]), float(c4.loc[idx, m+"_std"] or 0)

def get_cond(df, ds, model, cond):
    r = df[(df["dataset"]==ds) & (df["model"]==model) &
           (df["condition"]==cond)]
    if r.empty: return np.nan, 0.0
    task = r["task_type"].iloc[0]
    m = "f1_macro" if task=="classification" else "r2"
    return float(r[m].iloc[0]), float(r[m+"_std"].iloc[0] or 0)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE A  –  C0 vs C4 per ML model (4 subplots), all 7 datasets
# ─────────────────────────────────────────────────────────────────────────────
def figure_A(df):
    clf_ds = ["adult","bank","diabetes","heart","student"]
    reg_ds = ["life_expectancy","house_prices"]
    clf_models = ["logreg","rf","knn","gbm"]
    reg_models = ["ridge","rf","knn","gbm"]

    # One row per (dataset, model): C0, C4, task
    rows = []
    for ds in clf_ds + reg_ds:
        grp = df[df["dataset"]==ds]
        task = grp["task_type"].iloc[0]
        models = clf_models if task=="classification" else reg_models
        for model in models:
            c0v, c0s = get_cond(df, ds, model, "C0_raw")
            c4v, c4s = get_c4_best(df, ds, model)
            rows.append({"ds": ds, "model": model, "task": task,
                         "c0": c0v, "c0s": c0s, "c4": c4v, "c4s": c4s})
    data = pd.DataFrame(rows)

    all_models = clf_models  # ridge covered separately or treated as regression

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), sharey=False)

    for ax, model in zip(axes, clf_models):
        sub = data[data["model"]==model].copy()
        # Sort: clf datasets first, then reg (ridge not in clf models so sub will only have clf)
        sub = sub.sort_values("ds", key=lambda x: x.map(
            {d: i for i, d in enumerate(clf_ds+reg_ds)}))

        n = len(sub)
        x = np.arange(n)
        bw = 0.35

        bars_c0 = ax.bar(x - bw/2, sub["c0"].values, bw,
                         label="C0 (raw)", color=C["C0"],
                         yerr=sub["c0s"].values,
                         error_kw={"elinewidth":0.7,"capsize":2,"ecolor":"#555"},
                         zorder=3, alpha=0.9)
        bars_c4 = ax.bar(x + bw/2, sub["c4"].values, bw,
                         label="C4 (ours)", color=C["C4"],
                         yerr=sub["c4s"].values,
                         error_kw={"elinewidth":0.7,"capsize":2,"ecolor":"#555"},
                         zorder=3, alpha=0.9)

        # Annotate delta above C4 bar
        for i, row in sub.reset_index(drop=True).iterrows():
            delta = row["c4"] - row["c0"]
            thresh = row["c0s"] + row["c4s"]
            if abs(delta) > 0.0005:
                color = "#1b5e20" if delta > thresh else ("#b71c1c" if delta < -thresh else "#555")
                sign  = "+" if delta >= 0 else ""
                ypos  = max(row["c4"], row["c0"]) + max(row["c4s"], row["c0s"]) + 0.005
                ax.text(i + bw/2, ypos, f"{sign}{delta:.3f}",
                        ha="center", va="bottom", fontsize=6.5,
                        color=color, fontweight="bold" if abs(delta)>thresh else "normal")

        ax.set_xticks(x)
        ax.set_xticklabels([DS_LABEL.get(d, d) for d in sub["ds"].values],
                           rotation=30, ha="right")
        ax.set_title(MODEL_LABEL.get(model, model))
        ax.set_ylabel("f1_macro" if i == 0 else "")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        vals = list(sub["c0"].values) + list(sub["c4"].values)
        vals = [v for v in vals if v==v]
        if vals:
            ax.set_ylim(max(0, min(vals)-0.06), min(1.0, max(vals)+0.08))

    axes[0].legend(loc="lower right", framealpha=0.9)
    fig.suptitle("Figure A — C0 (raw baseline) vs C4 (agentic cleaning) per ML model across 7 datasets\n"
                 "Δ annotations: green=WIN (Δ > σ_C0+σ_C4), red=LOSS, grey=TIE",
                 fontsize=9, y=1.01)
    fig.tight_layout()
    for ext in ("pdf","png"):
        p = os.path.join(OUT, f"figureA_c0_vs_c4.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=200 if ext=="png" else 150)
    plt.close(fig)
    print(f"[OK] Figure A saved")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE B  –  C4 vs ALL external comparators, 4 common datasets
#             Shows best ML model score per system per dataset
# ─────────────────────────────────────────────────────────────────────────────
def figure_B(df):
    B_datasets = ["adult","bank","diabetes","heart"]

    df_flaml = pd.read_csv(FLAML)
    df_opt   = pd.read_csv(OPT)

    # C4 best per dataset
    def c4_best_ds(ds):
        r = df[(df["dataset"]==ds) & df["condition"].str.startswith("C4_")]
        m = "f1_macro"
        idx = r[m].idxmax()
        return float(r.loc[idx, m]), float(r.loc[idx, m+"_std"] or 0)

    # C0 best per dataset
    def c0_best_ds(ds):
        r = df[(df["dataset"]==ds) & (df["condition"]=="C0_raw")]
        m = "f1_macro"
        idx = r[m].idxmax()
        return float(r.loc[idx, m]), float(r.loc[idx, m+"_std"] or 0)

    # AutoML best per dataset
    def automl_best(df_ext, ds):
        r = df_ext[df_ext["dataset"]==ds]
        if r.empty: return np.nan, 0.0
        m = "f1_macro"
        idx = r[m].idxmax()
        return float(r.loc[idx, m]), float(r.loc[idx, m+"_std"] or 0)

    # Cloud LLM best per dataset (from MASTER, C6_*)
    def cloud_best(llm_tag, ds):
        cond = f"C6_{llm_tag}"
        r = df[(df["dataset"]==ds) & (df["condition"]==cond)]
        if r.empty: return np.nan, 0.0
        m = "f1_macro"
        idx = r[m].idxmax()
        return float(r.loc[idx, m]), float(r.loc[idx, m+"_std"] or 0)

    systems = ["C0", "C4", "FLAML", "Optuna", "ChatGPT", "Gemini", "Claude", "Copilot"]
    sys_colors = [C["C0"], C["C4"], C["FLAML"], C["Optuna"],
                  C["ChatGPT"], C["Gemini"], C["Claude"], C["Copilot"]]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), sharey=True)

    for ax, ds in zip(axes, B_datasets):
        vals, errs = [], []
        for sys in systems:
            if sys == "C0":
                v, e = c0_best_ds(ds)
            elif sys == "C4":
                v, e = c4_best_ds(ds)
            elif sys == "FLAML":
                v, e = automl_best(df_flaml, ds)
            elif sys == "Optuna":
                v, e = automl_best(df_opt, ds)
            elif sys == "ChatGPT":
                v, e = cloud_best("chatgpt", ds)
            elif sys == "Gemini":
                v, e = cloud_best("gemini", ds)
            elif sys == "Claude":
                v, e = cloud_best("claude", ds)
            elif sys == "Copilot":
                v, e = cloud_best("copilot", ds)
            vals.append(v); errs.append(e)

        x = np.arange(len(systems))
        bars = ax.bar(x, vals, 0.65, color=sys_colors,
                      yerr=errs,
                      error_kw={"elinewidth":0.8,"capsize":2,"ecolor":"#333"},
                      zorder=3, alpha=0.9)

        # Reference line at C4 score
        c4_score = vals[1]
        if not np.isnan(c4_score):
            ax.axhline(c4_score, color=C["C4"], lw=1.0, ls="--", alpha=0.7, zorder=2)

        # Annotate delta vs C4 for each bar
        for i, (v, e) in enumerate(zip(vals, errs)):
            if i == 1 or np.isnan(v) or np.isnan(c4_score): continue
            delta = v - c4_score
            if abs(delta) > 0.002:
                sign = "+" if delta >= 0 else ""
                # Check statistical significance vs C4
                c4_e = errs[1]
                thresh = e + c4_e
                is_sig = abs(delta) > thresh
                fc = "#b71c1c" if (delta < -thresh) else ("#1b5e20" if (delta > thresh) else "#555")
                ax.text(i, v + e + 0.004, f"{sign}{delta:.3f}",
                        ha="center", va="bottom", fontsize=5.5,
                        color=fc, fontweight="bold" if is_sig else "normal",
                        rotation=0)

        ax.set_xticks(x)
        ax.set_xticklabels(systems, rotation=40, ha="right")
        ax.set_title(DS_LABEL.get(ds, ds), fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("f1_macro (best ML model)")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Shared y range
    all_vals = [v for sys_idx in range(len(systems))
                for ds_idx, ds in enumerate(B_datasets)
                for v in [vals[sys_idx]] if not np.isnan(v)]

    # Legend patches
    patches = [mpatches.Patch(color=c, label=s)
               for s, c in zip(systems, sys_colors)]
    fig.legend(handles=patches, loc="lower center", ncol=8,
               bbox_to_anchor=(0.5, -0.04), framealpha=0.9, fontsize=7.5)

    fig.suptitle("Figure B — C4 vs external comparators across 4 common datasets\n"
                 "Dashed line = C4 score; Δ vs C4 annotated (bold = statistically significant)",
                 fontsize=9, y=1.01)
    fig.tight_layout()
    for ext in ("pdf","png"):
        p = os.path.join(OUT, f"figureB_comparators.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=200 if ext=="png" else 150)
    plt.close(fig)
    print(f"[OK] Figure B saved")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE C  –  Delta heatmap: C4 - C0 per dataset × ML model
# ─────────────────────────────────────────────────────────────────────────────
def figure_C(df):
    import matplotlib.cm as cm

    clf_ds = ["adult","bank","diabetes","heart","student"]
    reg_ds = ["life_expectancy","house_prices"]
    all_ds = clf_ds + reg_ds

    all_models = ["logreg","rf","knn","gbm","ridge"]

    matrix = np.full((len(all_models), len(all_ds)), np.nan)
    sig_matrix = np.zeros((len(all_models), len(all_ds)), dtype=bool)

    for j, ds in enumerate(all_ds):
        task = df[df["dataset"]==ds]["task_type"].iloc[0]
        for i, model in enumerate(all_models):
            c0v, c0s = get_cond(df, ds, model, "C0_raw")
            c4v, c4s = get_c4_best(df, ds, model)
            if np.isnan(c0v) or np.isnan(c4v): continue
            delta = c4v - c0v
            matrix[i, j] = delta
            if abs(delta) > c0s + c4s:
                sig_matrix[i, j] = True

    fig, ax = plt.subplots(figsize=(9, 3.8))

    vmax = np.nanmax(np.abs(matrix))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(range(len(all_ds)))
    ax.set_xticklabels([DS_LABEL.get(d,d) for d in all_ds], rotation=25, ha="right")
    ax.set_yticks(range(len(all_models)))
    ax.set_yticklabels([MODEL_LABEL.get(m,m) for m in all_models])

    # Annotate cells
    for i in range(len(all_models)):
        for j in range(len(all_ds)):
            v = matrix[i, j]
            if np.isnan(v): continue
            txt = f"{v:+.3f}"
            weight = "bold" if sig_matrix[i, j] else "normal"
            border = "★" if sig_matrix[i, j] else ""
            ax.text(j, i, f"{border}{txt}{border}",
                    ha="center", va="center", fontsize=7,
                    fontweight=weight, color="black")

    plt.colorbar(im, ax=ax, label="Δ (C4 − C0)", fraction=0.046, pad=0.04)
    ax.set_title("Figure C — C4 improvement over C0 baseline (Δ = C4 − C0)\n"
                 "★ = statistically significant (|Δ| > σ_C0 + σ_C4). Green = improvement, red = degradation.",
                 pad=8)
    fig.tight_layout()
    for ext in ("pdf","png"):
        p = os.path.join(OUT, f"figureC_delta_heatmap.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=200 if ext=="png" else 150)
    plt.close(fig)
    print(f"[OK] Figure C saved")


if __name__ == "__main__":
    df = load()
    figure_A(df)
    figure_B(df)
    figure_C(df)
    print("Done. All figures in:", OUT)
