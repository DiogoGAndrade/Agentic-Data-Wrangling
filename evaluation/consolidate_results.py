"""
Consolidate per-LLM result CSVs into a single master table.

Reads:    evaluation/outputs/results_<tag>.csv  (one per LLM)
Writes:   evaluation/outputs/MASTER_RESULTS_TABLE.csv
          evaluation/outputs/MASTER_PLAN_QUALITY.csv  (already managed by plan_quality.py)

Behaviour:
- Baselines (C0_raw, C1_manual) are recorded once (they are LLM-independent).
- C2_llm rows from each per-LLM file are stamped with the LLM tag and
  collapsed to condition='C2_<tag>' for direct comparison in the master.
- Both classification and regression metrics are preserved when present.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT_DIR = Path("evaluation/outputs")


def main() -> None:
    csv_files = sorted(OUT_DIR.glob("results_*.csv"))
    if not csv_files:
        print(f"[WARN] No results_*.csv files in {OUT_DIR}")
        return

    master_rows = []
    baselines_seen: set = set()  # track (dataset, condition, model) to deduplicate

    for fpath in csv_files:
        tag = fpath.stem.replace("results_", "")
        if tag in {"cv5"}:  # legacy filename
            continue

        df = pd.read_csv(fpath)
        if df.empty:
            continue

        for _, row in df.iterrows():
            cond = row.get("condition", "")
            row_dict = row.to_dict()

            if cond in {"C0_raw", "C1_manual"}:
                key = (row.get("dataset", ""), cond, row.get("model", ""))
                if key not in baselines_seen:
                    baselines_seen.add(key)
                    master_rows.append(row_dict)
            elif cond == "C2_llm":
                row_dict["condition"] = f"C2_{tag}"
                master_rows.append(row_dict)
            elif cond in ("C3_context", "C4_expanded"):
                # C3 files: results_c3_<llm_tag>_<ml_model>[_<dataset>].csv
                # C4 files: results_c4_<llm_tag>_<ml_model>[_<dataset>].csv
                cond_short = "c3" if cond == "C3_context" else "c4"
                cond_prefix_str = f"{cond_short}_"
                clean_tag = tag.replace(cond_prefix_str, "", 1) if tag.startswith(cond_prefix_str) else tag
                # Strip trailing _<dataset> suffix if present (added by fix to prevent overwrites)
                dataset_name = row.get("dataset", "")
                if dataset_name and clean_tag.endswith(f"_{dataset_name}"):
                    clean_tag = clean_tag[: -len(f"_{dataset_name}")]

                # Each CSV contains ALL ML models evaluated with ONE context plan.
                # Only keep the row where the ML model matches the context plan's target.
                # e.g. tag "qwen2.5_3b_logreg" → only keep model=logreg
                import re
                m = re.match(r'.+_(logreg|rf|knn|gbm|ridge)$', clean_tag)
                if m:
                    ctx_ml_model = m.group(1)
                    actual_model = row.get("model", "")
                    if actual_model != ctx_ml_model:
                        continue  # skip: wrong ML model for this context plan

                row_dict["condition"] = f"{cond_short.upper()}_{clean_tag}"
                master_rows.append(row_dict)
            else:
                # Pass-through for any other condition (e.g. legacy C2_iter1_naive)
                master_rows.append(row_dict)

    if not master_rows:
        print("[WARN] No rows to consolidate.")
        return

    master = pd.DataFrame(master_rows)

    # Deduplicate: keep last occurrence per (dataset, condition, model)
    # This handles the case where old CSVs (without dataset suffix) and new ones
    # (with dataset suffix) both exist and contain overlapping rows.
    dedup_cols = ["dataset", "condition", "model"]
    before = len(master)
    master = master.drop_duplicates(subset=dedup_cols, keep="last")
    if len(master) < before:
        print(f"[INFO] Removed {before - len(master)} duplicate rows")

    # Sort: dataset > model > condition (baselines first)
    cond_priority = {"C0_raw": 0, "C1_manual": 1}
    master["_cond_pri"] = master["condition"].map(lambda c: cond_priority.get(c, 9))
    master = master.sort_values(by=["dataset", "model", "_cond_pri", "condition"]).drop(columns="_cond_pri")

    # Round metric columns where present
    for col in ["accuracy", "f1_weighted", "f1_macro", "roc_auc",
                "mae", "rmse", "r2",
                "accuracy_std", "f1_weighted_std", "f1_macro_std", "roc_auc_std",
                "mae_std", "rmse_std", "r2_std"]:
        if col in master.columns:
            master[col] = pd.to_numeric(master[col], errors="coerce").round(4)

    out_path = OUT_DIR / "MASTER_RESULTS_TABLE.csv"
    master.to_csv(out_path, index=False)
    print(f"[OK] {out_path}  (rows={len(master)})")
    print(f"     Models consolidated: {[f.stem.replace('results_', '') for f in csv_files]}")


if __name__ == "__main__":
    main()
