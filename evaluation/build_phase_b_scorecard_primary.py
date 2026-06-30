"""
Phase B Scorecard — Primary Metric (f1_macro for classification, r2 for regression).
Reads MASTER_RESULTS_TABLE.csv, filters Phase B, calculates W/T/L with correct metrics.

Usage:
    python -m evaluation.build_phase_b_scorecard_primary
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

OUT_DIR = Path("evaluation/outputs")
PHASE_B_DATASETS = {"heart", "house_prices", "bank"}

PRIMARY_METRIC = {
    "classification": ("f1_macro", "f1_macro_std"),
    "regression": ("r2", "r2_std"),
}

ALL_METRICS = {
    "classification": [
        ("accuracy",    "accuracy_std",    1),
        ("f1_weighted", "f1_weighted_std",  1),
        ("f1_macro",    "f1_macro_std",     1),
        ("roc_auc",     "roc_auc_std",      1),
    ],
    "regression": [
        ("r2",   "r2_std",   1),
        ("mae",  "mae_std",  -1),   # lower is better → delta = C0 - C4
        ("rmse", "rmse_std", -1),
    ],
}


def wtl(delta: float, threshold: float) -> str:
    if delta > threshold:
        return "WIN"
    if delta < -threshold:
        return "LOSS"
    return "TIE"


def build_scorecard(df: pd.DataFrame, metric_col: str, std_col: str, direction: int = 1) -> pd.DataFrame:
    """direction=1 means higher is better, -1 means lower is better."""
    c0 = df[df["condition"] == "C0_raw"].set_index(["dataset", "model"])
    c4_rows = df[df["condition"].str.match(r"C4_qwen2\.5_(3b|14b)_(logreg|rf|knn|gbm|ridge)$")]

    rows = []
    for _, row in c4_rows.iterrows():
        dataset, model = row["dataset"], row["model"]
        cond = row["condition"]
        llm_tag = cond.replace(f"_{model}", "").replace("C4_", "")

        if (dataset, model) not in c0.index:
            continue
        c0r = c0.loc[(dataset, model)]

        if pd.isna(row.get(metric_col)) or pd.isna(c0r.get(metric_col)):
            continue

        c4_val = float(row[metric_col])
        c0_val = float(c0r[metric_col])
        c4_std = float(row.get(std_col, 0))
        c0_std = float(c0r.get(std_col, 0))

        raw_delta = (c4_val - c0_val) * direction  # positive = improvement
        threshold = c4_std + c0_std
        result = wtl(raw_delta, threshold)

        rows.append({
            "dataset": dataset,
            "llm": llm_tag,
            "model": model,
            "task_type": row.get("task_type", ""),
            "metric": metric_col,
            "C0": round(c0_val, 4),
            "C4": round(c4_val, 4),
            "delta": round(raw_delta, 4),
            "threshold": round(threshold, 4),
            "WTL": result,
        })

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, label: str) -> None:
    print(f"\n=== {label} ===")
    for llm in sorted(df["llm"].unique()):
        sub = df[df["llm"] == llm]
        w = (sub["WTL"] == "WIN").sum()
        t = (sub["WTL"] == "TIE").sum()
        l = (sub["WTL"] == "LOSS").sum()
        print(f"  {llm}: {w}W/{t}T/{l}L")
    wins = df[df["WTL"] == "WIN"]
    if not wins.empty:
        print("  WINs:")
        for _, r in wins.iterrows():
            print(f"    {r['dataset']}/{r['model']} [{r['metric']}] "
                  f"C0={r['C0']} C4={r['C4']} Δ={r['delta']:+.4f} thr={r['threshold']:.4f}")


def main() -> None:
    master = pd.read_csv(OUT_DIR / "MASTER_RESULTS_TABLE.csv")
    phase_b = master[master["dataset"].isin(PHASE_B_DATASETS)].copy()

    # ── PRIMARY SCORECARD ──────────────────────────────────────────────────
    primary_rows = []
    for task_type, (metric, std_col) in PRIMARY_METRIC.items():
        sub = phase_b[phase_b["task_type"] == task_type]
        chunk = build_scorecard(sub, metric, std_col, direction=1)
        primary_rows.append(chunk)

    primary = pd.concat(primary_rows, ignore_index=True)
    primary_path = OUT_DIR / "phase_b_scorecard_primary_corrected.csv"
    primary.to_csv(primary_path, index=False)
    print(f"[OK] {primary_path}")
    print_summary(primary, "Phase B — Primary Metric (f1_macro / r2)")

    # ── ALL-METRICS SCORECARD ──────────────────────────────────────────────
    all_rows = []
    for task_type, metric_list in ALL_METRICS.items():
        sub = phase_b[phase_b["task_type"] == task_type]
        for metric, std_col, direction in metric_list:
            chunk = build_scorecard(sub, metric, std_col, direction=direction)
            if not chunk.empty:
                all_rows.append(chunk)

    all_metrics = pd.concat(all_rows, ignore_index=True)
    all_path = OUT_DIR / "phase_b_scorecard_all_metrics_corrected.csv"
    all_metrics.to_csv(all_path, index=False)
    print(f"\n[OK] {all_path}")

    for llm in sorted(all_metrics["llm"].unique()):
        sub = all_metrics[all_metrics["llm"] == llm]
        w = (sub["WTL"] == "WIN").sum()
        t = (sub["WTL"] == "TIE").sum()
        l = (sub["WTL"] == "LOSS").sum()
        print(f"  {llm} all-metrics: {w}W/{t}T/{l}L")

    wins_all = all_metrics[all_metrics["WTL"] == "WIN"]
    if not wins_all.empty:
        print("  All-metrics WINs:")
        for _, r in wins_all.iterrows():
            print(f"    {r['dataset']}/{r['model']} [{r['metric']}] "
                  f"Δ={r['delta']:+.4f} thr={r['threshold']:.4f}")

    # ── AUDIT MARKDOWN ─────────────────────────────────────────────────────
    audit_path = OUT_DIR / "PHASE_B_SCORECARD_AUDIT.md"
    bank_rf = primary[(primary["dataset"] == "bank") & (primary["model"] == "rf") &
                      (primary["llm"] == "qwen2.5_3b")].iloc[0]

    md = f"""# Phase B Scorecard Audit

## Error Found

The original `phase_b_scorecard_final.csv` used **`accuracy`** as the primary metric
for classification datasets (heart, bank). The correct primary metric is **`f1_macro`**,
as defined in the thesis methodology and HANDOFF_CONTEXT.md.

| Item | Wrong | Correct |
|------|-------|---------|
| Classification primary metric | accuracy | f1_macro |
| Regression primary metric | r2 (correct) | r2 |

## Impact

`bank/rf` was scored as TIE (accuracy delta=+0.003, threshold=0.006).
With the correct metric (f1_macro), it is a **WIN** (delta=+0.023, threshold=0.018).

## New Scorecard — Primary Metric (f1_macro / r2)

| LLM | W | T | L |
|-----|---|---|---|
"""
    for llm in sorted(primary["llm"].unique()):
        sub = primary[primary["llm"] == llm]
        w = (sub["WTL"] == "WIN").sum()
        t = (sub["WTL"] == "TIE").sum()
        l = (sub["WTL"] == "LOSS").sum()
        md += f"| {llm} | {w} | {t} | {l} |\n"

    md += f"""
**Total Phase B C4: 2W / 22T / 0L** (1W per LLM × 2 LLMs)

## bank/rf Detail

| Field | Value |
|-------|-------|
| Dataset | bank |
| Model | rf |
| Metric | f1_macro |
| C0 f1_macro | {bank_rf['C0']} |
| C4 f1_macro | {bank_rf['C4']} |
| Delta | {bank_rf['delta']:+.4f} |
| Threshold (std_C0 + std_C4) | {bank_rf['threshold']:.4f} |
| WTL | **{bank_rf['WTL']}** |

## W/T/L Criterion

The criterion was NOT changed:
- WIN if Δ > std_C0 + std_C4
- LOSS if Δ < -(std_C0 + std_C4)
- TIE otherwise

Only the metric used to compute Δ was corrected.

## Files Changed

| File | Action |
|------|--------|
| `evaluation/build_phase_b_scorecard_primary.py` | Created (this script) |
| `evaluation/outputs/phase_b_scorecard_primary_corrected.csv` | Created |
| `evaluation/outputs/phase_b_scorecard_all_metrics_corrected.csv` | Created |
| `evaluation/outputs/PHASE_B_SCORECARD_AUDIT.md` | Created |
| `evaluation/outputs/phase_b_scorecard_final.csv` | Superseded (wrong metric) |
"""

    audit_path.write_text(md, encoding="utf-8")
    print(f"\n[OK] {audit_path}")


if __name__ == "__main__":
    main()
