"""Run Phase C evaluation with G8 active. Saves partial CSVs after each LLM."""
import sys, warnings, json, copy, time
sys.path.insert(0, '/sessions/gifted-focused-cannon/mnt/Projeto')
warnings.filterwarnings('ignore')

import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, r2_score
from sklearn.pipeline import Pipeline
from engine.cleaning_pipeline import PlanBasedCleaner
from evaluation.enforce_c4_v3 import enforce_plan, compute_cardinality
from evaluation.run_phase_c import (
    C4SafetyStep, encode_categoricals, PHASE_C, get_models,
    _MODEL_TO_DOWNSTREAM, N_FOLDS, EXPORTS, OUT
)
from engine.config import RANDOM_STATE

LOG = OUT / "phase_c_g8_progress.log"
OUT_CSV = OUT / "PHASE_C_RESULTS.csv"

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")

LLMS = [
    "qwen2.5:3b", "llama3.2:3b", "mistral:7b", "qwen2.5:7b",
    "llama3.1:8b", "gemma2:9b", "mistral-nemo:12b", "qwen2.5:14b",
]

def eval_all():
    all_rows = []
    c0_done = set()

    for llm in LLMS:
        llm_tag = llm.replace(":", "_").replace(".", "_")
        log(f"\n{'='*50}\nLLM: {llm}\n{'='*50}")
        t_llm = time.time()

        for dataset, spec in PHASE_C.items():
            c0_path = EXPORTS / dataset / "c0_raw.csv"
            plan_path = EXPORTS / dataset / "provenance" / f"c4_plan_{llm_tag}.json"
            if not c0_path.exists() or not plan_path.exists():
                log(f"  SKIP {dataset}")
                continue

            df = pd.read_csv(c0_path)
            plan_raw = json.loads(plan_path.read_text(encoding="utf-8"))
            target = spec["target"]
            task = spec["task"]
            card = compute_cardinality(df)
            X = df.drop(columns=[target], errors="ignore")
            y = df[target].copy()

            if task == "classification":
                cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                scoring = {"f1": make_scorer(f1_score, average="macro", zero_division=0)}
                mk = "test_f1"; mn = "f1_macro"
            else:
                cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                scoring = {"r2": make_scorer(r2_score)}
                mk = "test_r2"; mn = "r2"

            # C0 baseline (once per dataset)
            if dataset not in c0_done:
                X_c0 = encode_categoricals(X, target)
                X_c0 = X_c0.fillna(X_c0.median(numeric_only=True))
                for col in X_c0.select_dtypes(include='object').columns:
                    X_c0[col] = X_c0[col].astype('category').cat.codes
                for mname, model in get_models(task).items():
                    try:
                        s = cross_validate(Pipeline([("m",model)]), X_c0, y, cv=cv, scoring=scoring)
                        all_rows.append(dict(dataset=dataset,condition="C0",model=mname,task_type=task,
                                             n_folds=5,mean=round(float(np.mean(s[mk])),6),
                                             std=round(float(np.std(s[mk])),6),metric=mn))
                        log(f"  C0 {dataset} {mname}: {all_rows[-1]['mean']:.4f}")
                    except Exception as e:
                        log(f"  C0 {dataset} {mname} ERR: {e}")
                c0_done.add(dataset)

            # C4 per model
            for mname, model in get_models(task).items():
                try:
                    downstream = _MODEL_TO_DOWNSTREAM.get(mname, "RandomForest")
                    ctx = {**spec["user_context"], "downstream_model": downstream}
                    plan_e, ch = enforce_plan(copy.deepcopy(plan_raw), ctx, card)
                    g8_ch = [c for c in ch if 'impute' in c or 'cat_missing' in c]
                    pipe = Pipeline([
                        ("cl", PlanBasedCleaner(plan=plan_e, target_column=target)),
                        ("sf", C4SafetyStep()),
                        ("m", model),
                    ])
                    s = cross_validate(pipe, X, y, cv=cv, scoring=scoring, error_score="raise")
                    all_rows.append(dict(dataset=dataset,condition=f"C4_{llm_tag}",model=mname,
                                         task_type=task,n_folds=5,
                                         mean=round(float(np.mean(s[mk])),6),
                                         std=round(float(np.std(s[mk])),6),metric=mn))
                    log(f"  C4 {dataset} {mname}: {all_rows[-1]['mean']:.4f} G8={g8_ch}")
                except Exception as e:
                    log(f"  C4 {dataset} {mname} ERR: {e}")

        # Save partial after each LLM
        pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
        log(f"  [saved partial: {len(all_rows)} rows, llm_elapsed={time.time()-t_llm:.0f}s]")

    log(f"\n[DONE] {len(all_rows)} total rows")

if __name__ == "__main__":
    LOG.unlink(missing_ok=True)
    log("Starting Phase C G8 evaluation...")
    eval_all()
