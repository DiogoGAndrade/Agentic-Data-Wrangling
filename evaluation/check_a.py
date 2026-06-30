import pandas as pd
from pathlib import Path

root = Path(".")
datasets = ["adult", "diabetes", "student"]

for ds in datasets:
    base = root / "data" / "exports" / ds
    print(f"\n--- {ds.upper()} ---")

    c0 = pd.read_csv(base / "c0_raw.csv")
    c1 = pd.read_csv(base / "c1_manual.csv")
    c2 = pd.read_csv(base / "c2_llm.csv")

    print("c0_raw.csv   |", c0.shape)
    print("c1_manual.csv|", c1.shape)
    print("c2_llm.csv   |", c2.shape)

    same_cols = list(c0.columns) == list(c2.columns)
    same_shape = c0.shape == c2.shape
    print("C0 vs C2 same cols?", same_cols, "| same shape?", same_shape)

    cols_only_c0 = sorted(set(c0.columns) - set(c2.columns))
    cols_only_c2 = sorted(set(c2.columns) - set(c0.columns))

    if cols_only_c0:
        print("Columns removed in C2:", cols_only_c0)
    if cols_only_c2:
        print("Columns added in C2:", cols_only_c2)

    common_cols = [c for c in c0.columns if c in c2.columns]
    c0_common = c0[common_cols].astype(str)
    c2_common = c2[common_cols].astype(str)

    diffs_common = (c0_common != c2_common).sum().sum()
    print("C0 vs C2 different cells on common columns:", int(diffs_common))