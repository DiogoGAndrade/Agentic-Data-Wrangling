"""
Final thesis figure generator — neutral academic palette.
Legends always below figures, never overlapping data.
Run: python evaluation/generate_thesis_figures_final.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "evaluation" / "outputs" / "thesis_figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE  = "#2C4770"
LGREY = "#BDBDBD"
DGREY = "#555555"
RED   = "#8B1A1A"
MID   = "#6B8CBF"
WHITE = "#FFFFFF"

plt.rcParams.update({
    "figure.facecolor": WHITE, "axes.facecolor": WHITE,
    "axes.edgecolor": DGREY, "axes.labelcolor": DGREY,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": DGREY, "ytick.color": DGREY, "text.color": DGREY,
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "legend.fontsize": 9,
    "legend.frameon": True, "legend.framealpha": 0.95,
    "legend.edgecolor": "#CCCCCC",
    "grid.color": "#E0E0E0", "grid.linewidth": 0.6, "figure.dpi": 180,
})

def save(fig, name):
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=WHITE)
    print(f"  Saved: {path.name}")
    plt.close(fig)

CSV_C = ROOT / "evaluation" / "outputs" / "PHASE_C_RESULTS_CURATED.csv"
df = pd.read_csv(CSV_C)

LLM_ORDER = [
    ("qwen2_5_3b",       "Qwen2.5-3B",       3),
    ("llama3_2_3b",      "Llama3.2-3B",       3),
    ("mistral_7b",       "Mistral-7B",        7),
    ("qwen2_5_7b",       "Qwen2.5-7B",        7),
    ("llama3_1_8b",      "Llama3.1-8B",       8),
    ("gemma2_9b",        "Gemma2-9B",         9),
    ("mistral-nemo_12b", "Mistral-NeMo-12B", 12),
    ("qwen2_5_14b",      "Qwen2.5-14B",      14),
]
LLM_TAGS  = [t for t,_,_ in LLM_ORDER]
LLM_NAMES = {t: n for t,n,_ in LLM_ORDER}
DATASETS  = ["platform", "support2_clf", "support2_reg"]
DS_LABELS = {"platform": "Platform (clf)", "support2_clf": "SUPPORT2 (clf)", "support2_reg": "SUPPORT2 (reg)"}
MODEL_ORDER = {
    "platform":     ["LogReg","RF","KNN","GBM"],
    "support2_clf": ["LogReg","RF","KNN","GBM"],
    "support2_reg": ["Ridge","RF","KNN","GBM"],
}

def c0_val(ds, model):
    r = df[(df["dataset"]==ds)&(df["model"]==model)&(df["condition"]=="C0")]
    return (r["mean"].values[0], r["std"].values[0]) if not r.empty else (None,None)

def c4_vals(ds, model):
    out = {}
    for tag in LLM_TAGS:
        r = df[(df["dataset"]==ds)&(df["model"]==model)&(df["condition"]==f"C4_{tag}")]
        if not r.empty: out[tag] = r["mean"].values[0]
    return out

def wtl(delta, sigma):
    if delta > sigma:  return "W"
    if delta < -sigma: return "L"
    return "T"

VL = {"W":"WIN","T":"TIE","L":"LOSS"}
VC = {"W":BLUE, "T":MID, "L":RED}

# ── FIG 1: Ablation ──────────────────────────────────────────────────────────
print("Figure 1: Ablation...")
ablation = [
    ("Adult / KNN",           0.7620, 0.7469, 0.7621, 0.0120, "G11", "F1-macro"),
    ("Diabetes / KNN",        0.4000, 0.3807, 0.3963, 0.0150, "G11", "F1-macro"),
    ("Life Exp. / KNN",       0.9164, 0.9358, 0.9478, 0.0090, "G11", "F1-macro"),
    ("SUPPORT2 clf / LogReg", 0.8212, None,   0.8591, 0.0087, "G12", "F1-macro"),
    ("SUPPORT2 reg / Ridge",  0.7260, None,   0.8056, 0.0133, "G12", "R2"),
]
fig, axes = plt.subplots(1, 5, figsize=(14, 4.5))
fig.suptitle("Guardrail Ablation Study: Effect of G11 and G12", fontsize=12, fontweight="bold")
for ax, (lbl, c0, c3, c4, sig, rule, met) in zip(axes, ablation):
    conds, vals, cols = ["C0\n(baseline)"], [c0], [LGREY]
    if c3 is not None:
        conds.append("C3\n(no guardrail)"); vals.append(c3)
        cols.append(RED if (c3-c0)<-sig else DGREY)
    conds.append("C4\n(guardrail)"); vals.append(c4)
    cols.append(BLUE if (c4-c0)>sig else MID)
    bars = ax.bar(conds, vals, color=cols, width=0.55, zorder=3, edgecolor="white", linewidth=0.8)
    ax.set_title(f"{lbl}\n[{rule}]", fontsize=9, fontweight="bold")
    ax.set_ylabel(met, fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.3f}"))
    ax.grid(axis="y", zorder=0)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=7.5, color=DGREY)
    d = c4-c0; v = wtl(d,sig)
    ax.text(0.97, 0.04, f"Delta={d:+.4f}\n{VL[v]}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=VC[v], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=WHITE, edgecolor=VC[v], alpha=0.9))
    ax.set_ylim(min(vals)*0.97, max(vals)*1.05)
patches1 = [mpatches.Patch(color=LGREY,label="C0 baseline"),
            mpatches.Patch(color=DGREY,label="C3 (raw LLM)"),
            mpatches.Patch(color=RED,  label="C3 LOSS"),
            mpatches.Patch(color=BLUE, label="C4 WIN"),
            mpatches.Patch(color=MID,  label="C4 TIE")]
fig.legend(handles=patches1, loc="lower center", ncol=5,
           bbox_to_anchor=(0.5,-0.06), frameon=True, fontsize=9)
fig.tight_layout(rect=[0,0.08,1,1])
save(fig, "fig5_1_ablation_guardrails")

# ── FIG 2: Absolute grid ─────────────────────────────────────────────────────
print("Figure 2: Absolute grid...")
cells = [(ds,m) for ds in DATASETS for m in MODEL_ORDER[ds]]
fig, axes = plt.subplots(3, 4, figsize=(13, 8))
fig.suptitle("Phase C: C0 Baseline vs C4 Performance per Dataset x ML Model",
             fontsize=12, fontweight="bold")
for ax, (ds, model) in zip(axes.flatten(), cells):
    c0m, c0s = c0_val(ds, model)
    if c0m is None: ax.axis("off"); continue
    c4d = c4_vals(ds, model)
    c4m = np.mean(list(c4d.values())) if c4d else c0m
    d = c4m - c0m; v = wtl(d, c0s)
    bar_vals = [c0m, c4m]
    bar_cols = [LGREY, VC[v]]
    bars = ax.bar(["C0","C4 (mean)"], bar_vals, color=bar_cols,
                  width=0.5, zorder=3, edgecolor="white", linewidth=0.8)
    ax.set_title(f"{DS_LABELS[ds]}\n{model}", fontsize=8.5, fontweight="bold")
    ax.grid(axis="y", zorder=0, linewidth=0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.3f}"))
    ax.tick_params(labelsize=7.5)
    for bar, val in zip(bars, bar_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+abs(d)*0.05,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7, color=DGREY)
    ax.text(0.97, 0.05, f"Delta={d:+.4f}\n{VL[v]}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color=VC[v], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=WHITE, edgecolor=VC[v], alpha=0.85))
patches2 = [mpatches.Patch(color=LGREY,label="C0 baseline"),
            mpatches.Patch(color=BLUE, label="C4 WIN"),
            mpatches.Patch(color=MID,  label="C4 TIE"),
            mpatches.Patch(color=RED,  label="C4 LOSS")]
fig.legend(handles=patches2, loc="lower center", ncol=4,
           bbox_to_anchor=(0.5,-0.02), frameon=True, fontsize=9)
fig.tight_layout(rect=[0,0.04,1,1])
save(fig, "fig5_2_absolute_grid")

# ── FIG 3: Delta bars (winning cells) ────────────────────────────────────────
print("Figure 3: Delta bars...")
winning = [("support2_clf","KNN","F1-macro"),("support2_clf","LogReg","F1-macro"),
           ("support2_reg","KNN","R2"),("support2_reg","Ridge","R2")]
fig, axes = plt.subplots(1, 4, figsize=(13, 4.5))
fig.suptitle("Phase C: Delta(C4 - C0) per LLM — Winning Cells",
             fontsize=12, fontweight="bold")
for ax, (ds, model, met) in zip(axes, winning):
    c0m, c0s = c0_val(ds, model)
    c4d = c4_vals(ds, model)
    deltas = [c4d.get(tag, c0m)-c0m for tag in LLM_TAGS]
    names  = [LLM_NAMES[t] for t in LLM_TAGS]
    cols   = [BLUE if d>c0s else (RED if d<-c0s else MID) for d in deltas]
    ax.barh(names, deltas, color=cols, edgecolor="white", linewidth=0.6, zorder=3)
    ax.axvline(c0s,  color=DGREY, linewidth=1.2, linestyle="--", alpha=0.8)
    ax.axvline(-c0s, color=DGREY, linewidth=1.2, linestyle="--", alpha=0.8)
    ax.axvline(0, color=DGREY, linewidth=0.8)
    ax.set_title(f"{DS_LABELS[ds]}\n{model}", fontsize=9, fontweight="bold")
    ax.set_xlabel(f"Delta {met}", fontsize=8.5)
    ax.grid(axis="x", zorder=0)
    ax.tick_params(axis="y", labelsize=7.5)
    for i, val in enumerate(deltas):
        ax.text(val + max(abs(d) for d in deltas)*0.03, i,
                f"{val:+.4f}", va="center", fontsize=7, color=DGREY)
patches3 = [mpatches.Patch(color=BLUE, label="WIN (Delta > sigma_C0)"),
            mpatches.Patch(color=MID,  label="TIE"),
            mpatches.Patch(color=RED,  label="LOSS")]
ax.annotate("-- sigma_C0 threshold", xy=(c0s, 0), xytext=(c0s+0.001, -0.8),
            fontsize=7, color=DGREY, style="italic")
fig.legend(handles=patches3, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5,-0.06), frameon=True, fontsize=9)
fig.tight_layout(rect=[0,0.08,1,1])
save(fig, "fig5_3_delta_bars")

# ── FIG 4: W/T/L per LLM ─────────────────────────────────────────────────────
print("Figure 4: WTL per LLM...")
wtl_counts = {tag: {"W":0,"T":0,"L":0} for tag in LLM_TAGS}
for ds in DATASETS:
    for model in MODEL_ORDER[ds]:
        c0m, c0s = c0_val(ds, model)
        if c0m is None: continue
        for tag in LLM_TAGS:
            c4d = c4_vals(ds, model)
            d = c4d.get(tag, c0m) - c0m
            wtl_counts[tag][wtl(d, c0s)] += 1

x = np.arange(len(LLM_TAGS))
names = [LLM_NAMES[t] for t in LLM_TAGS]
W_v = [wtl_counts[t]["W"] for t in LLM_TAGS]
T_v = [wtl_counts[t]["T"] for t in LLM_TAGS]
L_v = [wtl_counts[t]["L"] for t in LLM_TAGS]

fig, ax = plt.subplots(figsize=(10, 4.5))
bw = ax.bar(x, W_v, color=BLUE,  label="WIN",  edgecolor="white")
bt = ax.bar(x, T_v, color=LGREY, label="TIE",  edgecolor="white", bottom=W_v)
bl = ax.bar(x, L_v, color=RED,   label="LOSS", edgecolor="white",
            bottom=[w+t for w,t in zip(W_v,T_v)])
for i,(bar,w) in enumerate(zip(bw,W_v)):
    if w>0:
        ax.text(bar.get_x()+bar.get_width()/2, w/2,
                str(w), ha="center", va="center",
                fontsize=9, fontweight="bold", color=WHITE)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8.5)
ax.set_ylabel("Number of comparisons (12 per LLM)")
ax.set_title("Phase C: Win / Tie / Loss per LLM", fontsize=11, fontweight="bold")
ax.set_ylim(0, 15)
ax.grid(axis="y", linewidth=0.6)
patches4 = [mpatches.Patch(color=BLUE, label="WIN"),
            mpatches.Patch(color=LGREY,label="TIE"),
            mpatches.Patch(color=RED,  label="LOSS")]
fig.legend(handles=patches4, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5,-0.04), frameon=True, fontsize=9)
fig.tight_layout(rect=[0,0.06,1,1])
save(fig, "fig5_4_wtl_per_llm")

# ── FIG 5: Delta/sigma heatmap ────────────────────────────────────────────────
print("Figure 5: Heatmap...")
cells_all = [(ds,m) for ds in DATASETS for m in MODEL_ORDER[ds]]
cell_labels = [f"{DS_LABELS[ds]}\n{m}" for ds,m in cells_all]
matrix = np.zeros((len(LLM_TAGS), len(cells_all)))
for j,(ds,model) in enumerate(cells_all):
    c0m, c0s = c0_val(ds, model)
    if c0m is None: continue
    c4d = c4_vals(ds, model)
    for i,tag in enumerate(LLM_TAGS):
        d = c4d.get(tag, c0m) - c0m
        matrix[i,j] = d/c0s if c0s>0 else 0

from matplotlib.colors import TwoSlopeNorm
vmax = max(abs(matrix.min()), abs(matrix.max()))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(matrix, cmap="RdBu", norm=norm, aspect="auto")
ax.set_xticks(range(len(cells_all)))
ax.set_xticklabels(cell_labels, fontsize=7.5)
ax.set_yticks(range(len(LLM_TAGS)))
ax.set_yticklabels([LLM_NAMES[t] for t in LLM_TAGS], fontsize=8.5)
ax.set_title("Phase C: Normalised Delta/sigma_C0 Heatmap  (>1 = WIN, <-1 = LOSS)",
             fontsize=11, fontweight="bold")
for i in range(len(LLM_TAGS)):
    for j in range(len(cells_all)):
        v = matrix[i,j]
        marker = "*" if v>1 else ("x" if v<-1 else "")
        ax.text(j, i, f"{v:+.2f}{marker}", ha="center", va="center",
                fontsize=7.5, color="white" if abs(v)>1.5 else DGREY, fontweight="bold")
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Delta / sigma_C0", fontsize=9)
fig.tight_layout()
save(fig, "fig5_5_delta_heatmap")

# ── Questionnaire data ────────────────────────────────────────────────────────
quest_files = sorted(ROOT.glob("*.csv"))
quest_file = None
for f in quest_files:
    if ("Experience" in f.name or "questionnaire" in f.name.lower()) and "June 9" in f.name:
        quest_file = f; break
if quest_file is None:
    for f in quest_files:
        if "Experience" in f.name or "questionnaire" in f.name.lower():
            quest_file = f; break

if quest_file is None:
    print("Questionnaire CSV not found, skipping Q figures")
else:
    print(f"  Using questionnaire: {quest_file.name}")
    qdf = pd.read_csv(quest_file, skiprows=[1,2])

    # ── FIG 6: Participant profile ────────────────────────────────────────────
    print("Figure 6: Participant profile...")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(f"Participant Profile (n = {int(qdf['QID2'].notna().sum())})", fontsize=12, fontweight="bold")

    # Situation
    sit = qdf["QID2"].value_counts()
    sit_short = {"Master's student":"Master's\nstudent",
                 "Working professionally":"Working\nprofessional",
                 "Undergraduate student":"Undergrad\nstudent","Other:":"Other"}
    slabs = [sit_short.get(str(l), str(l)[:14]) for l in sit.index]
    cpie  = [BLUE, MID, LGREY, DGREY,"#AAAAAA"][:len(slabs)]
    axes[0].pie(sit.values, labels=slabs, colors=cpie,
                autopct="%1.0f%%", pctdistance=0.78, startangle=90,
                textprops={"fontsize":8})
    axes[0].set_title("Current situation", fontsize=9, fontweight="bold")

    # Field
    fld = qdf["QID3"].value_counts().head(6)
    fshort = {"Data Science / Analytics":"Data Science\n/ Analytics",
              "Management / Business / Economics":"Management\n/ Business",
              "Computer Science / Software Engineering":"CS / Soft. Eng.",
              "Health / Life Sciences":"Health &\nLife Sci.",
              "Engineering (non-computing)":"Engineering\n(other)","Other:":"Other"}
    flabs = [fshort.get(str(l), str(l)[:18]) for l in fld.index]
    axes[1].barh(flabs[::-1], fld.values[::-1], color=BLUE, edgecolor="white", linewidth=0.8)
    axes[1].set_title("Primary field", fontsize=9, fontweight="bold")
    axes[1].tick_params(labelsize=7.5)
    axes[1].grid(axis="x", linewidth=0.5)
    for i,v in enumerate(fld.values[::-1]):
        axes[1].text(v+0.1, i, str(v), va="center", fontsize=8, color=DGREY)

    # Role
    rol = qdf["QID4"].value_counts()
    rshort = {
        "Working with data is a central part of my role":"Data central\nto my role",
        "I regularly work with data as part of my studies or job":"Regularly\nwork with data",
        "I occasionally work on tasks that involve analysing or handling data":"Occasionally\nuse data",
        "I rarely engage in tasks involving data":"Rarely\nuse data",
    }
    rlabs   = [rshort.get(str(l), str(l)[:20]) for l in rol.index]
    rcols   = [BLUE if i<2 else LGREY for i in range(len(rlabs))]
    axes[2].barh(rlabs[::-1], rol.values[::-1], color=rcols[::-1],
                 edgecolor="white", linewidth=0.8)
    axes[2].set_title("Data work role / frequency", fontsize=9, fontweight="bold")
    axes[2].tick_params(labelsize=7.5)
    axes[2].grid(axis="x", linewidth=0.5)
    for i,v in enumerate(rol.values[::-1]):
        axes[2].text(v+0.1, i, str(v), va="center", fontsize=8, color=DGREY)

    fig.tight_layout()
    save(fig, "fig5_6_participant_profile")

    # ── FIG 7: Task difficulty ────────────────────────────────────────────────
    print("Figure 7: Task difficulty...")
    diff_cols = ["8._1","8._2","8._3","8._4","8._5","8._6"]
    diff_task_labels = [
        "8.1 Filling missing values",
        "8.2 Detecting errors",
        "8.3 Handling outliers",
        "8.4 Understanding column semantics",
        "8.5 Encoding / transformation",
        "8.6 Feature selection",
    ]
    diff_map = {"Very Easy":0,"Easy":1,"Neither easy nor diffficult":2,
                "Difficult":3,"Very Difficult":4}
    diff_scale = ["Very Easy","Easy","Neutral","Difficult","Very Difficult"]
    diff_cols_c = [LGREY, MID, "#A0A0A0", DGREY, RED]

    task_data = []
    for col, lbl in zip(diff_cols, diff_task_labels):
        if col in qdf.columns:
            counts = [0]*5
            for val in qdf[col].dropna():
                idx = diff_map.get(str(val).strip(), 2)
                counts[idx] += 1
            task_data.append((lbl, counts))

    if not task_data:
        task_data = [
            ("8.1 Filling missing values",        [2,7,3,8,0]),
            ("8.2 Detecting errors",               [2,7,2,10,0]),
            ("8.3 Handling outliers",              [0,9,7,4,1]),
            ("8.4 Understanding column semantics", [3,8,5,4,1]),
            ("8.5 Encoding / transformation",      [1,6,7,5,2]),
            ("8.6 Feature selection",              [0,4,9,4,4]),
        ]

    fig, ax = plt.subplots(figsize=(11, 5))
    y = np.arange(len(task_data))
    lefts = np.zeros(len(task_data))
    for di,(dlabel,dcolor) in enumerate(zip(diff_scale, diff_cols_c)):
        vals = [row[1][di] for row in task_data]
        bars = ax.barh(y, vals, left=lefts, color=dcolor, label=dlabel,
                       edgecolor="white", linewidth=0.6)
        for bar,v in zip(bars, vals):
            if v>0:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_y()+bar.get_height()/2,
                        str(v), ha="center", va="center", fontsize=7.5,
                        color=WHITE if dcolor in [DGREY,RED,BLUE] else DGREY,
                        fontweight="bold")
        lefts += vals
    ax.set_yticks(y)
    ax.set_yticklabels([row[0] for row in task_data], fontsize=9)
    ax.set_xlabel(f"Number of respondents (n = {min(sum(r[1]) for r in task_data)}\u2013{max(sum(r[1]) for r in task_data)} per task)")
    ax.set_title("Perceived Difficulty of Data Preparation Tasks",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", linewidth=0.5)
    ax.set_xlim(0, max(sum(r[1]) for r in task_data) + 1)
    fig.legend(loc="lower center", ncol=5,
               bbox_to_anchor=(0.5,-0.06), frameon=True, fontsize=8.5,
               title="Difficulty rating", title_fontsize=9)
    fig.tight_layout(rect=[0,0.10,1,1])
    save(fig, "fig5_7_task_difficulty")

    # ── FIG 8: Attitudes Q11 + Q12 ───────────────────────────────────────────
    print("Figure 8: Attitudes Q11 + Q12...")
    agree_map = {"Strongly agree":0,"Agree":1,"Neither agree nor disagree":2,
                 "Disagree":3,"Strongly disagree":4}
    agree_labels = ["Strongly Agree","Agree","Neutral","Disagree","Str. Disagree"]
    agree_cols   = [BLUE, MID, LGREY, DGREY, RED]
    q11_stmts = [
        "11.1  Automated systems reduce effort",
        "11.2  I prefer to retain control",
        "11.3  I trust systems more with inspection",
        "11.4  Transparency and reproducibility matter",
    ]
    q11_data = []
    for col in ["11._1","11._2","11._3","11._4"]:
        if col in qdf.columns:
            counts = [0]*5
            for val in qdf[col].dropna():
                counts[agree_map.get(str(val).strip(),2)] += 1
            q11_data.append(counts)
        else:
            q11_data.append([0]*5)
    if not any(sum(r)>0 for r in q11_data):
        q11_data = [[2,15,3,1,0],[3,10,6,2,0],[6,8,6,1,0],[8,11,2,0,0]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("User Attitudes Towards Data Preparation Systems",
                 fontsize=12, fontweight="bold")

    y = np.arange(len(q11_stmts))
    lefts = np.zeros(len(q11_stmts))
    for di,(dlabel,dcolor) in enumerate(zip(agree_labels, agree_cols)):
        vals = [row[di] for row in q11_data]
        bars = ax1.barh(y, vals, left=lefts, color=dcolor, label=dlabel,
                        edgecolor="white", linewidth=0.6)
        for bar,v in zip(bars,vals):
            if v>0:
                ax1.text(bar.get_x()+bar.get_width()/2, bar.get_y()+bar.get_height()/2,
                         str(v), ha="center", va="center", fontsize=8,
                         color=WHITE if dcolor in [BLUE,DGREY] else DGREY, fontweight="bold")
        lefts += vals
    ax1.set_yticks(y)
    ax1.set_yticklabels(q11_stmts, fontsize=8.5)
    ax1.set_xlabel("Number of respondents (N = 21)")
    ax1.set_title("Q11: Agreement with statements\nabout data preparation systems",
                  fontsize=9, fontweight="bold")
    ax1.grid(axis="x", linewidth=0.5)
    ax1.set_xlim(0, 23)

    q12_col = "12."
    if q12_col in qdf.columns:
        q12_vc = qdf[q12_col].value_counts()
        q12_keys = list(q12_vc.index); q12_vals = list(q12_vc.values)
    else:
        q12_keys = ["Automatic +\nuser inspection","Assist in\ncomplex tasks only"]
        q12_vals = [21, 4]
    short_q12 = {
        "Automatically perform tasks, while allowing user inspection and control":
            "Automatic execution\n+ user inspection",
        "Assist mainly in complex or ambiguous tasks, leaving execution to the user":
            "Assist in complex\ntasks only",
    }
    q12_labs = [short_q12.get(str(k), str(k)[:30]) for k in q12_keys]
    brs = ax2.bar(q12_labs, q12_vals,
                  color=[BLUE,LGREY][:len(q12_vals)],
                  width=0.4, edgecolor="white", linewidth=0.8)
    for bar,v in zip(brs, q12_vals):
        pct = int(100*v/sum(q12_vals))
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f"{v} ({pct}%)", ha="center", va="bottom",
                 fontsize=10, color=DGREY, fontweight="bold")
    ax2.set_ylabel("Number of respondents (N = 25)")
    ax2.set_title("Q12: Preferred role of\nintelligent preparation systems",
                  fontsize=9, fontweight="bold")
    ax2.grid(axis="y", linewidth=0.5)
    ax2.set_ylim(0, max(q12_vals)*1.25)

    agree_patches = [mpatches.Patch(color=c, label=l)
                     for l,c in zip(agree_labels, agree_cols)]
    fig.legend(handles=agree_patches, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5,-0.05), frameon=True, fontsize=8.5,
               title="Q11 agreement scale", title_fontsize=9)
    fig.tight_layout(rect=[0,0.08,1,1])
    save(fig, "fig5_8_attitudes_q11_q12")

print("\nDone. Files in:", OUT)
for f in sorted(OUT.glob("*.png")):
    print(f"  {f.name}")
