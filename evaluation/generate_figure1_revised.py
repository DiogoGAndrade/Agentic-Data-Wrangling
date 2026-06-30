"""
Revised Figure 1 — Two-panel comparison for thesis Chapter 5.

Panel A: C0 vs C1 vs C4-best, all 7 datasets, best ML model per dataset.
Panel B: C4 vs FLAML vs Optuna vs ChatGPT vs Claude, 4 common datasets.

Run: python evaluation/generate_figure1_revised.py
Output: evaluation/outputs/figures/
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# ── colour palette ──────────────────────────────────────────────────────────
C = {
    "C0":      "#9e9e9e",   # grey
    "C1":      "#f57c00",   # amber
    "C4":      "#1565C0",   # blue
    "FLAML":   "#2e7d32",   # green
    "Optuna":  "#66bb6a",   # light green
    "ChatGPT": "#e65100",   # orange
    "Claude_zs": "#b71c1c", # dark red
}

# ── data helpers ─────────────────────────────────────────────────────────────
def load_master(path):
    df = pd.read_csv(path)
    df["condition"] = df["condition"].fillna("")
    return df

def best_per_dataset(df, prefix):
    """Return best score (across ML models) per dataset for a condition prefix."""
    sub = df[df["condition"].str.startswith(prefix)]
    out = {}
    for ds, grp in sub.groupby("dataset"):
        task = grp["task_type"].iloc[0]
        metric = "f1_macro" if task == "classification" else "r2"
        std_col = metric + "_std"
        idx = grp[metric].idxmax()
        out[ds] = {
            "score": float(grp.loc[idx, metric]),
            "std":   float(grp.loc[idx, std_col]) if pd.notna(grp.loc[idx, std_col]) else 0.0,
            "task":  task,
            "model": grp.loc[idx, "model"],
        }
    return out

def exact_condition(df, cond):
    sub = df[df["condition"] == cond]
    out = {}
    for ds, grp in sub.groupby("dataset"):
        task = grp["task_type"].iloc[0]
        metric = "f1_macro" if task == "classification" else "r2"
        std_col = metric + "_std"
        idx = grp[metric].idxmax()
        out[ds] = {
            "score": float(grp.loc[idx, metric]),
            "std":   float(grp.loc[idx, std_col]) if pd.notna(grp.loc[idx, std_col]) else 0.0,
        }
    return out

# ── Panel A helper: best model per dataset (use C0's best model for fair comparison) ──
def best_model_per_dataset_trio(df):
    """For each dataset, pick the ML model where C4 is highest, then get C0/C1/C4 for that model."""
    records = []
    for ds, grp in df.groupby("dataset"):
        task = grp["task_type"].iloc[0]
        metric = "f1_macro" if task == "classification" else "r2"
        std_col = metric + "_std"

        c4 = grp[grp["condition"].str.startswith("C4_")]
        if c4.empty:
            continue
        best_idx = c4[metric].idxmax()
        best_model = c4.loc[best_idx, "model"]

        def get_val(cond_prefix, exact=False):
            if exact:
                rows = grp[(grp["condition"] == cond_prefix) & (grp["model"] == best_model)]
            else:
                rows = grp[grp["condition"].str.startswith(cond_prefix) & (grp["model"] == best_model)]
            if rows.empty:
                return np.nan, 0.0
            idx2 = rows[metric].idxmax()
            v = rows.loc[idx2, metric]
            s = rows.loc[idx2, std_col]
            return float(v), float(s) if pd.notna(s) else 0.0

        c0_v, c0_s = get_val("C0_raw", exact=True)
        c1_v, c1_s = get_val("C1_manual", exact=True)
        c4_v, c4_s = get_val("C4_")

        records.append({
            "dataset": ds, "task": task, "model": best_model,
            "C0": c0_v, "C0_std": c0_s,
            "C1": c1_v, "C1_std": c1_s,
            "C4": c4_v, "C4_std": c4_s,
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
base   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
master = os.path.join(base, "MASTER_RESULTS_TABLE.csv")
df     = load_master(master)

# External comparator CSVs
df_flaml  = pd.read_csv(os.path.join(base, "results_flaml.csv"))
df_optuna = pd.read_csv(os.path.join(base, "results_optuna.csv"))
df_cloud  = pd.read_csv(os.path.join(base, "results_cloud_llm_all.csv"))

# ── Panel A data ──────────────────────────────────────────────────────────────
trio = best_model_per_dataset_trio(df)

# Dataset display order: clf first, reg after
clf_ds = ["adult", "bank", "diabetes", "heart", "student"]
reg_ds = ["life_expectancy", "house_prices"]
label_map = {
    "adult": "Adult", "bank": "Bank", "diabetes": "Diabetes",
    "heart": "Heart", "student": "Student",
    "life_expectancy": "Life Exp.\n(R²)", "house_prices": "House\nPrices (R²)",
}
ds_order_A = [d for d in clf_ds + reg_ds if d in trio["dataset"].values]
trio = trio.set_index("dataset").loc[ds_order_A].reset_index()

# ── Panel B data ──────────────────────────────────────────────────────────────
# 4 common datasets: adult, bank, diabetes, heart
B_datasets = ["adult", "bank", "diabetes", "heart"]

c4_best = best_per_dataset(df, "C4_")

def automl_best(df_ext, sys_col="system"):
    out = {}
    for _, row in df_ext.iterrows():
        ds = row["dataset"]
        score = row["f1_macro"] if row["task_type"] == "classification" else row["r2"]
        std   = row["f1_macro_std"] if row["task_type"] == "classification" else row["r2_std"]
        if pd.notna(score):
            out[ds] = {"score": float(score), "std": float(std) if pd.notna(std) else 0.0}
    return out

flaml_d  = automl_best(df_flaml)
optuna_d = automl_best(df_optuna)

cloud_best = {}
for llm, grp in df_cloud.groupby("llm_tag"):
    for ds, ds_grp in grp.groupby("dataset"):
        idx  = ds_grp["f1_macro"].idxmax()
        best = ds_grp.loc[idx, "f1_macro"]
        std  = ds_grp.loc[idx, "f1_macro_std"]
        cloud_best.setdefault(llm, {})[ds] = {
            "score": float(best),
            "std":   float(std) if pd.notna(std) else 0.0,
        }

B_systems = ["C4", "FLAML", "Optuna", "ChatGPT", "Claude z.s."]
B_lookup  = {
    "C4":         {ds: c4_best.get(ds, {}) for ds in B_datasets},
    "FLAML":      flaml_d,
    "Optuna":     optuna_d,
    "ChatGPT":    cloud_best.get("chatgpt", {}),
    "Claude z.s.": cloud_best.get("claude", {}),
}

# ── FIGURE ────────────────────────────────────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(
    1, 2, figsize=(14, 5),
    gridspec_kw={"width_ratios": [7, 4], "wspace": 0.35}
)

bar_w = 0.25

# ── PANEL A ───────────────────────────────────────────────────────────────────
n_A   = len(ds_order_A)
offA  = [-bar_w, 0, bar_w]
x_A   = np.arange(n_A)

for i, (cond, col) in enumerate([("C0","C0"),("C1","C1"),("C4","C4")]):
    vals = trio[cond].values
    errs = trio[cond + "_std"].values
    ax_a.bar(x_A + offA[i], vals, bar_w,
             label=cond, color=C[col],
             yerr=errs, error_kw={"elinewidth":0.8,"capsize":2,"ecolor":"#444"},
             zorder=3, alpha=0.9)

# Annotate WIN ticks on C4 bar
for i, row in trio.iterrows():
    ds = row["dataset"]
    c0, c0s = row["C0"], row["C0_std"]
    c4, c4s = row["C4"], row["C4_std"]
    delta = c4 - c0
    threshold = c0s + c4s
    if delta > threshold:
        ypos = c4 + c4s + 0.003
        ax_a.text(i + offA[2], ypos, "WIN", ha="center", va="bottom",
                  fontsize=6.5, color="#1565C0", fontweight="bold")

# Separator clf / reg
n_clf = sum(1 for d in ds_order_A if d in clf_ds)
ax_a.axvline(n_clf - 0.5, color="#888", lw=0.8, ls=":", alpha=0.6)
ax_a.text(n_clf - 0.45, ax_a.get_ylim()[0] if ax_a.get_ylim()[0] > 0 else 0.3,
          "← f1_macro | R² →", fontsize=6.5, color="#666", va="bottom")

ax_a.set_xticks(x_A)
ax_a.set_xticklabels([label_map.get(d, d) for d in ds_order_A])
ax_a.set_ylabel("Score (f1_macro or R²)")
ax_a.set_title("(A)  Baseline progression: C0 → C1 → C4\n(best ML model per dataset; error bars = ±1 std, 5-fold CV)")
ax_a.legend(loc="lower right", framealpha=0.9)

all_A = list(trio["C0"].values) + list(trio["C1"].values) + list(trio["C4"].values)
ax_a.set_ylim(max(0, min(all_A) - 0.04), min(1.0, max(all_A) + 0.06))

# ── PANEL B ───────────────────────────────────────────────────────────────────
n_B   = len(B_datasets)
n_Bsys = len(B_systems)
bar_w_B = 0.14
offB = np.linspace(-(n_Bsys - 1) / 2 * bar_w_B, (n_Bsys - 1) / 2 * bar_w_B, n_Bsys)
x_B  = np.arange(n_B)

color_map_B = {"C4": C["C4"], "FLAML": C["FLAML"], "Optuna": C["Optuna"],
               "ChatGPT": C["ChatGPT"], "Claude z.s.": C["Claude_zs"]}

for i, sys in enumerate(B_systems):
    vals = []
    errs = []
    for ds in B_datasets:
        entry = B_lookup[sys].get(ds, {})
        vals.append(entry.get("score", np.nan))
        errs.append(entry.get("std", 0.0))
    vals = np.array(vals, dtype=float)
    errs = np.array(errs, dtype=float)
    ax_b.bar(x_B + offB[i], vals, bar_w_B,
             label=sys, color=color_map_B[sys],
             yerr=errs, error_kw={"elinewidth":0.8,"capsize":2,"ecolor":"#444"},
             zorder=3, alpha=0.9)

# Mark Claude LOSS on bank
# Bank is index 1
bank_idx = B_datasets.index("bank")
claude_bank = B_lookup["Claude z.s."].get("bank", {}).get("score", np.nan)
c4_bank     = B_lookup["C4"].get("bank", {}).get("score", np.nan)
if not np.isnan(claude_bank):
    ax_b.annotate("LOSS", xy=(bank_idx + offB[4], claude_bank),
                  xytext=(bank_idx + offB[4] - 0.05, claude_bank - 0.04),
                  fontsize=6.5, color="#b71c1c", fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color="#b71c1c", lw=0.8))

ax_b.set_xticks(x_B)
ax_b.set_xticklabels([label_map.get(d, d) for d in B_datasets])
ax_b.set_ylabel("f1_macro")
ax_b.set_title("(B)  C4 vs external comparators\n(4 common datasets; cloud LLMs zero-shot, no guardrails)")
ax_b.legend(loc="lower right", framealpha=0.9, ncol=2)

all_B = [B_lookup[s].get(ds, {}).get("score", np.nan)
         for s in B_systems for ds in B_datasets]
all_B = [v for v in all_B if not np.isnan(v)]
ax_b.set_ylim(max(0, min(all_B) - 0.08), min(1.0, max(all_B) + 0.06))

# ── Save ──────────────────────────────────────────────────────────────────────
fig.suptitle("Figure 1 — Performance of C4 across conditions and comparator systems\n"
             "(5-fold cross-validation, RANDOM_STATE=369)",
             fontsize=10, y=1.01)

fig.tight_layout()
pdf_path = os.path.join(OUT_DIR, "figure1_revised.pdf")
png_path = os.path.join(OUT_DIR, "figure1_revised.png")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, bbox_inches="tight", dpi=200)
plt.close(fig)
print("[OK] Figure 1 revised PDF: " + pdf_path)
print("[OK] Figure 1 revised PNG: " + png_path)
