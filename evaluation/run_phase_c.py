"""Phase C — High-Missingness Stress Test
Runs the full C4 pipeline for platform, support2_clf, support2_reg.
Usage: python evaluation/run_phase_c.py --step all --all-models
"""
import argparse, copy, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.cleaning_pipeline import PlanBasedCleaner
from engine.config import RANDOM_STATE
from engine.profile_dataset import build_dataset_profile
from llm.ollama_client import OllamaClient
from evaluation.prepare_conditions import build_plan_prompt_c4
from evaluation.enforce_c4_v3 import enforce_plan, compute_cardinality
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score, r2_score
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import warnings
warnings.filterwarnings("ignore")

OLLAMA_URL = "http://localhost:11434"
EXPORTS = ROOT / "data" / "exports"
OUT = ROOT / "evaluation" / "outputs"
OUT_FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

PHASE_C = {
    "platform": {
        "target": "purchased",
        "task": "classification",
        "leakage_cols": [],
        "user_context": {
            "downstream_model": "RandomForest",
            "column_semantics": {
                "age": "numeric", "income": "numeric",
                "days_on_platform": "numeric",
                "gender": "categorical", "city": "categorical",
            },
            "redundant_features": [], "leakage_cols": [],
        },
    },
    "support2_clf": {
        "target": "hospdead",
        "task": "classification",
        "leakage_cols": [],
        "user_context": {
            "downstream_model": "RandomForest",
            "column_semantics": {
                "age": "numeric", "meanbp": "numeric", "hrt": "numeric",
                "resp": "numeric", "temp": "numeric",
                "sps": "numeric", "aps": "numeric",
            },
            "redundant_features": [], "leakage_cols": [],
        },
    },
    "support2_reg": {
        "target": "log_charges",
        "task": "regression",
        "leakage_cols": [],
        "user_context": {
            "downstream_model": "RandomForest",
            "column_semantics": {
                "age": "numeric", "meanbp": "numeric", "hrt": "numeric",
            },
            "redundant_features": [], "leakage_cols": [],
        },
    },
}

LLM_ALL = [
    "qwen2.5:3b", "llama3.2:3b", "mistral:7b", "qwen2.5:7b",
    "llama3.1:8b", "gemma2:9b", "mistral-nemo:12b", "qwen2.5:14b",
]
N_FOLDS = 5

# Map sklearn model names to downstream_model for enforce_plan.
# Linear/KNN -> one_hot encoding; trees -> ordinal encoding.
_MODEL_TO_DOWNSTREAM = {
    "LogReg": "LogisticRegression",
    "Ridge":  "Ridge",
    "KNN":    "KNN",
    "RF":     "RandomForest",
    "GBM":    "GradientBoosting",
}


def generate_plan(df, dataset, llm):
    spec = PHASE_C[dataset]
    target = spec["target"]
    ctx = spec["user_context"]
    client = OllamaClient(base_url=OLLAMA_URL, model=llm)
    profile = build_dataset_profile(df, target_column=target)
    prompt = build_plan_prompt_c4(
        dataset_name=dataset,
        columns=list(df.columns),
        preview_rows=df.head(8).to_dict(orient="records"),
        target_column=target,
        dataset_profile=profile,
        aggressive_filter=False,
        user_context=ctx,
    )
    print(f"    [LLM] {llm} generating plan for {dataset}...", end=" ", flush=True)
    try:
        raw = client.generate(prompt).strip()
        start, end = raw.find("{"), raw.rfind("}") + 1
        plan = json.loads(raw[start:end]) if start != -1 else {"actions": []}
        print(f"OK ({len(plan.get('actions', []))} actions)")
        return plan
    except Exception as e:
        print(f"FAILED ({e})")
        return {"actions": []}


def get_models(task):
    if task == "classification":
        return {
            "LogReg": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "RF":     RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
            "KNN":    KNeighborsClassifier(n_neighbors=5),
            "GBM":    GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        }
    else:
        return {
            "Ridge": Ridge(random_state=RANDOM_STATE),
            "RF":    RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
            "KNN":   KNeighborsRegressor(n_neighbors=5),
            "GBM":   GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
        }


def encode_categoricals(df, target):
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        if col == target:
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


class C4SafetyStep(BaseEstimator, TransformerMixin):
    """Guardrail 7: ensures numeric, NaN-free output after PlanBasedCleaner.
    Handles residual object columns and missing values. Stateful - no leakage."""
    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy()
        df = df.where(df.notnull(), other=np.nan)
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        self._obj_cols = obj_cols
        if obj_cols:
            df[obj_cols] = df[obj_cols].fillna("__missing__").astype(str)
            self._enc = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1,
                encoded_missing_value=-1,
            )
            self._enc.fit(df[obj_cols])
            df[obj_cols] = self._enc.transform(df[obj_cols])
        else:
            self._enc = None
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        num_nan_cols = [c for c in df.columns if df[c].isna().any()]
        self._num_nan_cols = num_nan_cols
        if num_nan_cols:
            self._imp = SimpleImputer(strategy="median")
            self._imp.fit(df[num_nan_cols])
        else:
            self._imp = None
        self._fit_columns = list(df.columns)
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(X).copy()
        df = df.where(df.notnull(), other=np.nan)
        if self._enc is not None:
            cols_present = [c for c in self._obj_cols if c in df.columns]
            if cols_present:
                df[cols_present] = self._enc.transform(
                    df[cols_present].fillna("__missing__").astype(str)
                )
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if self._imp is not None:
            cols_present = [c for c in self._num_nan_cols if c in df.columns]
            if cols_present:
                df[cols_present] = self._imp.transform(df[cols_present])
        df = df.fillna(0)
        return df.values


def evaluate_condition(df, plan_raw, dataset, condition, user_context_base):
    """5-fold CV for all models. C4 uses model-aware enforcement per model.

    enforce_plan mutates the plan dict in-place. deepcopy is mandatory before
    each call so each model receives an independent copy of the original plan.
    Without deepcopy, the second model's enforce_plan call overwrites what the
    first set, making all models share the last model's encoding strategy.
    """
    spec = PHASE_C[dataset]
    target = spec["target"]
    task = spec["task"]
    cardinality = compute_cardinality(df)

    X = df.drop(columns=[target], errors="ignore")
    y = df[target].copy()

    if task == "classification":
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scoring = {"f1_macro": make_scorer(f1_score, average="macro", zero_division=0)}
        main_metric = "test_f1_macro"
    else:
        cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scoring = {"r2": make_scorer(r2_score)}
        main_metric = "test_r2"

    results = []
    for model_name, model in get_models(task).items():
        try:
            if condition == "C0":
                X_enc = encode_categoricals(X, target)
                X_enc = X_enc.fillna(X_enc.median(numeric_only=True))
                for col in X_enc.select_dtypes(include="object").columns:
                    X_enc[col] = X_enc[col].astype("category").cat.codes
                pipe = Pipeline([("model", model)])
                scores = cross_validate(pipe, X_enc, y, cv=cv, scoring=scoring, error_score="raise")
            else:
                # Model-aware enforcement: linear/KNN get one_hot, trees get ordinal.
                # deepcopy is mandatory - enforce_plan mutates the plan dict in-place.
                downstream = _MODEL_TO_DOWNSTREAM.get(model_name, "RandomForest")
                ctx = {**user_context_base, "downstream_model": downstream}
                plan_enforced, changes = enforce_plan(copy.deepcopy(plan_raw), ctx, cardinality)
                if changes:
                    print(
                        "        [" + model_name + " G] "
                        + str(len(changes)) + " changes: "
                        + str(changes[:2]) + ("..." if len(changes) > 2 else "")
                    )
                pipe = Pipeline([
                    ("cleaner", PlanBasedCleaner(plan=plan_enforced, target_column=target)),
                    ("safety", C4SafetyStep()),
                    ("model", model),
                ])
                scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, error_score="raise")

            vals = scores[main_metric]
            row = {
                "dataset": dataset,
                "condition": condition,
                "model": model_name,
                "task_type": task,
                "n_folds": N_FOLDS,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "metric": "f1_macro" if task == "classification" else "r2",
            }
            print("      " + model_name + ": "
                  + str(round(row["mean"], 4)) + " +/- " + str(round(row["std"], 4)))
            results.append(row)
        except Exception as e:
            print("      " + model_name + ": ERROR -- " + str(e))
    return results


def step_prepare(llm):
    for dataset, spec in PHASE_C.items():
        ds_dir = EXPORTS / dataset
        c0_path = ds_dir / "c0_raw.csv"
        if not c0_path.exists():
            print(f"[SKIP] {dataset}: c0_raw.csv not found at {c0_path}")
            continue
        print(f"\n[PREPARE] {dataset}")
        df = pd.read_csv(c0_path)
        plan = generate_plan(df, dataset, llm)
        llm_tag = llm.replace(":", "_").replace(".", "_")
        prov_dir = ds_dir / "provenance"
        prov_dir.mkdir(exist_ok=True)
        plan_path = prov_dir / ("c4_plan_" + llm_tag + ".json")
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(f"    [OK] Plan saved: {plan_path}")


def step_evaluate(llm):
    all_results = []
    llm_tag = llm.replace(":", "_").replace(".", "_")
    for dataset, spec in PHASE_C.items():
        ds_dir = EXPORTS / dataset
        c0_path = ds_dir / "c0_raw.csv"
        if not c0_path.exists():
            print(f"[SKIP] {dataset}: c0_raw.csv not found")
            continue
        plan_path = ds_dir / "provenance" / ("c4_plan_" + llm_tag + ".json")
        if not plan_path.exists():
            print(f"[WARN] {dataset}: plan not found. Run --step prepare first.")
            continue
        df = pd.read_csv(c0_path)
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        user_context = spec["user_context"]
        print(f"\n[EVALUATE] {dataset} | C0 baseline")
        c0_results = evaluate_condition(df, {}, dataset, "C0", user_context)
        all_results.extend(c0_results)
        print(f"\n[EVALUATE] {dataset} | C4 ({llm})")
        c4_results = evaluate_condition(df, plan, dataset, "C4_" + llm_tag, user_context)
        all_results.extend(c4_results)
    return all_results


def plot_phase_c(df_results):
    datasets = df_results["dataset"].unique()
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(5.5 * n_ds, 5), sharey=False)
    if n_ds == 1:
        axes = [axes]
    COLORS = {"C0": "#9e9e9e", "C4": "#1565C0"}
    for ax, ds in zip(axes, datasets):
        sub = df_results[df_results["dataset"] == ds]
        models = sub["model"].unique()
        x = np.arange(len(models))
        bw = 0.35
        c0 = sub[sub["condition"] == "C0"].set_index("model")
        c4_cond = ([c for c in sub["condition"].unique() if c.startswith("C4")] or [None])[0]
        ax.bar(x - bw/2, [c0.loc[m, "mean"] if m in c0.index else 0 for m in models],
               bw, label="C0 (raw)", color=COLORS["C0"], alpha=0.9)
        if c4_cond:
            c4 = sub[sub["condition"] == c4_cond].set_index("model")
            ax.bar(x + bw/2, [c4.loc[m, "mean"] if m in c4.index else 0 for m in models],
                   bw, label="C4 (LLM)", color=COLORS["C4"], alpha=0.9)
            for i, m in enumerate(models):
                if m in c0.index and m in c4.index:
                    delta = c4.loc[m, "mean"] - c0.loc[m, "mean"]
                    color = "#1b5e20" if delta > 0 else "#b71c1c"
                    sign = "+" if delta >= 0 else ""
                    ax.text(i + bw/2, c4.loc[m, "mean"] + 0.005,
                            sign + str(round(delta, 3)),
                            ha="center", va="bottom", fontsize=6.5,
                            color=color, fontweight="bold")
        spec = PHASE_C[ds]
        metric_label = "F1-macro" if spec["task"] == "classification" else "R2"
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylabel(metric_label)
        ax.set_title(ds + "\n(target: " + spec["target"] + ")", fontsize=9)
        ax.legend(fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.suptitle(
        "Phase C - C0 vs C4 on High-Missingness Datasets\n"
        "(platform: 48.9% missing; support2: ICU clinical data)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = OUT_FIG / ("phase_c_c0_vs_c4." + ext)
        fig.savefig(p, bbox_inches="tight", dpi=200 if ext == "png" else 150)
        print(f"[OK] {p}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase C - High-Missingness Stress Test")
    parser.add_argument("--step", choices=["prepare", "evaluate", "figures", "all"],
                        default="all")
    parser.add_argument("--llm", default="qwen2.5:3b")
    parser.add_argument("--all-models", action="store_true")
    args = parser.parse_args()
    llms = LLM_ALL if args.all_models else [args.llm]

    # Accumulate results across all LLMs before writing CSV once at the end.
    all_rows = []
    for llm in llms:
        print("\n" + "="*60 + "\nLLM: " + llm + "\n" + "="*60)
        if args.step in ("prepare", "all"):
            step_prepare(llm)
        if args.step in ("evaluate", "all"):
            all_rows.extend(step_evaluate(llm))

    if args.step in ("evaluate", "all") and all_rows:
        out_path = OUT / "PHASE_C_RESULTS.csv"
        df_r = pd.DataFrame(all_rows)
        df_r.to_csv(out_path, index=False)
        print("\n[OK] Results saved: " + str(out_path))
        print(df_r.to_string(index=False))

    if args.step in ("figures", "all"):
        results_path = OUT / "PHASE_C_RESULTS.csv"
        if results_path.exists():
            df_r = pd.read_csv(results_path)
            plot_phase_c(df_r)
        else:
            print("[WARN] No results file found. Run --step evaluate first.")
