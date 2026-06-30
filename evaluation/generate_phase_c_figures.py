"""
Phase C Figure Generator — uses evaluation/outputs/PHASE_C_RESULTS.csv
Generates:
  phase_c_absolute_grid  — absolute scores C0 vs C4 best, by dataset × model
  phase_c_delta_bars     — delta (C4-C0) per LLM, grouped by dataset/model
  phase_c_scaling        — mean delta vs LLM size (scaling law)
  phase_c_wtl_per_llm    — W/T/L stacked bars per LLM

Run: python evaluation/generate_phase_c_figures.py
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
CSV  = ROOT / "evaluation" / "outputs" / "PHASE_C_RESULTS.csv"
OUT  = ROOT / "evaluation" / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── LLM metadata ───────────────────────────────────────────────────────────────
LLM_ORDER = [
    ("qwen2_5_3b",      "Qwen2.5-3B",      3),
    ("llama3_2_3b",     "Llama3.2-3B",     3),
    ("mistral_7b",      "Mistral-7B",      7),
    ("qwen2_5_7b",      "Qwen2.5-7B",      7),
    ("llama3_1_8b",     "Llama3.1-8B",     8),
    ("gemma2_9b",       "Gemma2-9B",       9),
    ("mistral-nemo_12b","Mistral-NeMo-12B",12),
    ("qwen2_5_14b",     "Qwen2.5-14B",    14),
]
LLM_TAGS  = [t for t,_,_ in LLM_ORDER]
LLM_NAMES = {t: n for t,n,_ in LLM_ORDER}
LLM_SIZES = {t: s for t,_,s in LLM_ORDER}

DATASETS = ["platform", "support2_clf", "support2_reg"]
DS_LABELS = {
    "platform":     "Platform\n(clf)",
    "support2_clf": "Support2\n(clf)",
    "support2_reg": "Support2\n(reg)",
}

MODEL_ORDER = {
    "platform":     ["LogReg","RF","KNN","GBM"],
    "support2_clf": ["LogReg","RF","KNN","GBM"],
    "support2_reg": ["Ridge","RF","KNN","GBM"],
}

C = {
    "C0":  "#607D8B",
    "C4":  "#1565C0",
    "win": "#2E7D32",
    "tie": "#F9A825",
    "loss":"#C62828",
}

METRIC = {"platform":"f1_macro","support2_clf":"f1_macro","support2_reg":"r2"}

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV)

def c0_row(ds, model):
    r = df[(df["dataset"]==ds)&(df["condition"]=="C0")&(df["model"]==model)]
    if r.empty: return None, None
    return float(r["mean"].iloc[0]), float(r["std"].iloc[0])

def c4_best(ds, model):
    r = df[(df["dataset"]==ds)&(df["model"]==model)&(df["condition"].str.startswith("C4_"))]
    if r.empty: return None, None
    idx = r["mean"].idxmax()
    return float(r.loc[idx,"mean"]), float(r.loc[idx,"std"])

def wtl(ds, model, llm_tag):
    c0m, c0s = c0_row(ds, model)
    if c0m is None: return None
    r = df[(df["dataset"]==ds)&(df["model"]==model)&(df["condition"]==f"C4_{llm_tag}")]
    if r.empty: return None
    delta = float(r["mean"].iloc[0]) - c0m
    if delta > c0s:  return "W"
    if delta < -c0s: return "L"
    return "T"


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Absolute grid: C0 vs C4 best per dataset × model
# ══════════════════════════════════════════════════════════════════════════════
def fig_absolute_grid():
    nds = len(DATASETS)
    fig, axes = plt.subplots(1, nds, figsize=(4.5*nds, 4.5), sharey=False)
    plt.rcParams.update({"font.size":9})

    for ax, ds in zip(axes, DATASETS):
        models = MODEL_ORDER[ds]
        nm = len(models)
        x = np.arange(nm)
        bw = 0.35

        c0_means, c0_stds = [], []
        c4_means, c4_stds = [], []
        for m in models:
            v, s = c0_row(ds, m)
            c0_means.append(v if v else 0); c0_stds.append(s if s else 0)
            v, s = c4_best(ds, m)
            c4_means.append(v if v else 0); c4_stds.append(s if s else 0)

        bars0 = ax.bar(x - bw/2, c0_means, bw, label="C0 (raw)", color=C["C0"],
                       yerr=c0_stds, capsize=3, error_kw={"linewidth":0.8})
        bars4 = ax.bar(x + bw/2, c4_means, bw, label="C4 (ours)", color=C["C4"],
                       yerr=c4_stds, capsize=3, error_kw={"linewidth":0.8})

        # delta annotations
        for i, (c0v, c4v, c0s) in enumerate(zip(c0_means, c4_means, c0_stds)):
            delta = c4v - c0v
            color = C["win"] if delta > c0s else (C["loss"] if delta < -c0s else C["tie"])
            top = max(c0v, c4v) + max(c0_stds[i], c4_stds[i]) + 0.012
            ax.text(x[i] + bw/2, top, f"{delta:+.3f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color=color)

        metric_label = "F1-macro" if METRIC[ds]=="f1_macro" else "R²"
        ax.set_ylabel(metric_label, fontsize=9)
        ax.set_title(DS_LABELS[ds], fontsize=10, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8.5)
        ax.set_ylim(bottom=max(0, min(c0_means+c4_means) - 0.08))
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        if ds == DATASETS[0]:
            ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Phase C: C0 (raw) vs C4 (LLM-guided cleaning) — best LLM per cell",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(OUT/f"phase_c_absolute_grid.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] phase_c_absolute_grid")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Delta bars: C4−C0 per LLM, for each winning cell
# ══════════════════════════════════════════════════════════════════════════════
def fig_delta_bars():
    # Only show cells where at least one LLM wins
    winning_cells = []
    for ds in DATASETS:
        for m in MODEL_ORDER[ds]:
            c0m, c0s = c0_row(ds, m)
            if c0m is None: continue
            deltas = []
            for tag in LLM_TAGS:
                r = df[(df["dataset"]==ds)&(df["model"]==m)&(df["condition"]==f"C4_{tag}")]
                if not r.empty:
                    deltas.append(float(r["mean"].iloc[0]) - c0m)
            if any(d > c0s for d in deltas):
                winning_cells.append((ds, m))

    if not winning_cells:
        # Show all cells
        winning_cells = [(ds,m) for ds in DATASETS for m in MODEL_ORDER[ds]]

    nc = len(winning_cells)
    fig, axes = plt.subplots(1, nc, figsize=(3.2*nc, 4.0), sharey=False)
    if nc == 1: axes = [axes]
    plt.rcParams.update({"font.size":9})

    for ax, (ds, m) in zip(axes, winning_cells):
        c0m, c0s = c0_row(ds, m)
        tags = LLM_TAGS
        names = [LLM_NAMES[t] for t in tags]
        deltas = []
        for tag in tags:
            r = df[(df["dataset"]==ds)&(df["model"]==m)&(df["condition"]==f"C4_{tag}")]
            deltas.append(float(r["mean"].iloc[0]) - c0m if not r.empty else 0)

        colors = [C["win"] if d > c0s else (C["loss"] if d < -c0s else C["tie"]) for d in deltas]
        y = np.arange(len(tags))
        ax.barh(y, deltas, color=colors, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axvline(c0s, color=C["win"], linewidth=1.2, linestyle="--", alpha=0.7, label=f"σ={c0s:.4f}")
        ax.axvline(-c0s, color=C["loss"], linewidth=1.2, linestyle="--", alpha=0.7)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.5)
        ax.set_xlabel("Δ vs C0", fontsize=8.5)
        ax.set_title(f"{DS_LABELS[ds].replace(chr(10),' ')}\n{m}", fontsize=9, fontweight="bold")
        ax.xaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Phase C: Δ(C4 − C0) per LLM — winning cells", fontsize=11, fontweight="bold")
    plt.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(OUT/f"phase_c_delta_bars.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] phase_c_delta_bars")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Scaling: mean delta vs LLM param count
# ══════════════════════════════════════════════════════════════════════════════
def fig_scaling():
    # Compute mean delta per LLM across all dataset×model cells
    rows = []
    for tag, name, size in LLM_ORDER:
        deltas_all = []
        for ds in DATASETS:
            for m in MODEL_ORDER[ds]:
                c0m, c0s = c0_row(ds, m)
                if c0m is None: continue
                r = df[(df["dataset"]==ds)&(df["model"]==m)&(df["condition"]==f"C4_{tag}")]
                if not r.empty:
                    deltas_all.append(float(r["mean"].iloc[0]) - c0m)
        if deltas_all:
            rows.append({"tag":tag,"name":name,"size":size,
                         "mean_delta":np.mean(deltas_all),
                         "std_delta":np.std(deltas_all),
                         "n_wins":sum(1 for d in deltas_all if d > 0),
                         "n_cells":len(deltas_all)})

    sdf = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 4))
    plt.rcParams.update({"font.size":9})

    ax.errorbar(sdf["size"], sdf["mean_delta"]*100,
                yerr=sdf["std_delta"]*100,
                fmt="o-", color=C["C4"], markersize=7,
                capsize=4, linewidth=1.5, elinewidth=1)

    for _, row in sdf.iterrows():
        ax.annotate(row["name"], (row["size"], row["mean_delta"]*100),
                    textcoords="offset points", xytext=(4, 4), fontsize=7.5)

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("LLM Parameter Count (B)", fontsize=10)
    ax.set_ylabel("Mean Δ over C0 (×100, pp)", fontsize=10)
    ax.set_title("Phase C: LLM size vs mean performance delta over C0", fontsize=11, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(OUT/f"phase_c_scaling.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] phase_c_scaling")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — W/T/L per LLM stacked bar
# ══════════════════════════════════════════════════════════════════════════════
def fig_wtl_per_llm():
    results = []
    for tag, name, size in LLM_ORDER:
        W=T=L=0
        for ds in DATASETS:
            for m in MODEL_ORDER[ds]:
                r = wtl(ds, m, tag)
                if r == "W": W+=1
                elif r == "T": T+=1
                elif r == "L": L+=1
        results.append({"name":name,"W":W,"T":T,"L":L,"size":size})

    rdf = pd.DataFrame(results)
    n = len(rdf)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    plt.rcParams.update({"font.size":9})

    total = rdf["W"]+rdf["T"]+rdf["L"]
    p_w = ax.bar(x, rdf["W"], color=C["win"],  label="WIN",  edgecolor="white")
    p_t = ax.bar(x, rdf["T"], bottom=rdf["W"], color=C["tie"], label="TIE", edgecolor="white")
    p_l = ax.bar(x, rdf["L"], bottom=rdf["W"]+rdf["T"], color=C["loss"], label="LOSS", edgecolor="white")

    for i, row in rdf.iterrows():
        tot = row["W"]+row["T"]+row["L"]
        ax.text(i, tot + 0.2, f'W={row["W"]}', ha="center", fontsize=8, fontweight="bold", color=C["win"])

    ax.set_xticks(x)
    ax.set_xticklabels(rdf["name"], rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("# dataset×model cells", fontsize=10)
    ax.set_title("Phase C: Win/Tie/Loss per LLM across all dataset×model cells", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
    ax.set_ylim(0, max(total)+3)

    # Secondary x-axis with size annotation
    for i, row in rdf.iterrows():
        ax.text(i, -1.8, f'{row["size"]}B', ha="center", fontsize=7.5, color="grey")
    ax.text(-0.6, -1.8, "params:", ha="left", fontsize=7.5, color="grey")

    plt.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(OUT/f"phase_c_wtl_per_llm.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] phase_c_wtl_per_llm")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Delta heatmap: LLM × dataset/model cell
# ══════════════════════════════════════════════════════════════════════════════
def fig_delta_heatmap():
    cells = [(ds, m) for ds in DATASETS for m in MODEL_ORDER[ds]]
    cell_labels = [f"{DS_LABELS[ds].replace(chr(10),' ')}/{m}" for ds,m in cells]
    llm_names = [LLM_NAMES[t] for t in LLM_TAGS]

    mat = np.zeros((len(LLM_TAGS), len(cells)))
    sig_mat = np.zeros((len(LLM_TAGS), len(cells)))  # 1=win, -1=loss, 0=tie

    for j, (ds, m) in enumerate(cells):
        c0m, c0s = c0_row(ds, m)
        if c0m is None: continue
        for i, tag in enumerate(LLM_TAGS):
            r = df[(df["dataset"]==ds)&(df["model"]==m)&(df["condition"]==f"C4_{tag}")]
            if not r.empty:
                d = float(r["mean"].iloc[0]) - c0m
                mat[i, j] = d * 100  # convert to pp
                if d > c0s:    sig_mat[i,j] =  1
                elif d < -c0s: sig_mat[i,j] = -1

    vmax = max(abs(mat.min()), abs(mat.max()), 0.5)
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(len(cells)*0.9+1.5, len(LLM_TAGS)*0.6+1.5))
    plt.rcParams.update({"font.size":9})

    im = ax.imshow(mat, cmap="RdYlGn", norm=norm, aspect="auto")

    # annotate cells
    for i in range(len(LLM_TAGS)):
        for j in range(len(cells)):
            txt = f"{mat[i,j]:+.2f}"
            marker = "★" if sig_mat[i,j]==1 else ("✗" if sig_mat[i,j]==-1 else "")
            color = "white" if abs(mat[i,j]) > vmax*0.5 else "black"
            ax.text(j, i, f"{txt}\n{marker}", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold" if sig_mat[i,j]!=0 else "normal")

    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(cell_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(LLM_TAGS)))
    ax.set_yticklabels(llm_names, fontsize=8.5)
    ax.set_title("Phase C: Δ(C4−C0) heatmap (×100 pp) — ★=WIN, ✗=LOSS", fontsize=11, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Δ (×100 pp)", fontsize=9)

    plt.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(OUT/f"phase_c_delta_heatmap.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] phase_c_delta_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Loading {CSV}")
    print(f"Rows: {len(df)}  |  Saving to {OUT}")
    print()
    fig_absolute_grid()
    fig_delta_bars()
    fig_scaling()
    fig_wtl_per_llm()
    fig_delta_heatmap()
    print("\nAll figures saved.")
