"""Generate all C4 context spec JSON files for Phase A datasets."""
import json
from pathlib import Path

OUT = Path(__file__).parent

# ========================================================================
# ADULT dataset column semantics
# ========================================================================
ADULT_SEMANTICS = {
    "workclass": "nominal — employment sector, no natural order",
    "education": "ordinal — ordered education level (Preschool < ... < Doctorate)",
    "marital_status": "nominal — marital category, no order",
    "occupation": "nominal — job type, no order",
    "relationship": "nominal — family role, no order",
    "race": "nominal — racial category, no order",
    "sex": "nominal — binary gender, no order",
    "native_country": "nominal — country of origin, no order (high cardinality: 41 values)",
}
ADULT_REDUNDANT = ["education_num"]  # education_num is numeric encoding of education
ADULT_DESC = (
    "UCI Census Income dataset. Predicts whether annual income exceeds $50K based on "
    "demographic features. education_num is a numeric re-encoding of the education column "
    "and should be dropped as redundant. fnlwgt is a census sampling weight, not a predictor."
)

# ========================================================================
# DIABETES dataset column semantics  (after leakage cols dropped)
# ========================================================================
DIABETES_SEMANTICS = {
    "race": "nominal — racial category, no order",
    "gender": "nominal — gender, no order",
    "age": "ordinal — age bracket [0-10) < [10-20) < ... < [90-100)",
    "weight": "ordinal — weight bracket [0-25) < [25-50) < ... (mostly missing)",
    "payer_code": "nominal — insurance type, no order",
    "medical_specialty": "nominal — medical specialty, no order (73 unique, high cardinality)",
    "max_glu_serum": "ordinal — glucose level (Norm < >200 < >300)",
    "A1Cresult": "ordinal — HbA1c result (Norm < >7 < >8)",
    "metformin": "ordinal — medication change (No < Steady < Down < Up)",
    "repaglinide": "ordinal — medication change (No < Steady < Down < Up)",
    "nateglinide": "ordinal — medication change",
    "chlorpropamide": "ordinal — medication change",
    "glimepiride": "ordinal — medication change",
    "glipizide": "ordinal — medication change",
    "glyburide": "ordinal — medication change",
    "pioglitazone": "ordinal — medication change",
    "rosiglitazone": "ordinal — medication change",
    "acarbose": "ordinal — medication change",
    "miglitol": "ordinal — medication change",
    "insulin": "ordinal — medication change",
    "glyburide-metformin": "ordinal — medication change",
    "change": "nominal — binary flag (Ch/No)",
    "diabetesMed": "nominal — binary flag (Yes/No)",
    "diag_1": "nominal — ICD-9 code, high cardinality (717 unique)",
    "diag_2": "nominal — ICD-9 code, high cardinality",
    "diag_3": "nominal — ICD-9 code, high cardinality",
}
DIABETES_REDUNDANT = ["examide", "citoglipton"]  # single-value columns
DIABETES_DESC = (
    "Diabetes hospital readmission prediction. 130 US hospitals, 1999-2008. "
    "Predicts whether a patient will be readmitted within 30 days. "
    "Many medication columns with ordinal semantics (No < Steady < Down < Up). "
    "examide and citoglipton have only 1 unique value and are zero-variance."
)

# ========================================================================
# STUDENT dataset column semantics  (after leakage cols dropped)
# ========================================================================
STUDENT_SEMANTICS = {
    "school": "nominal — school identifier (GP/MS)",
    "sex": "nominal — gender (F/M)",
    "address": "nominal — urban/rural (U/R)",
    "famsize": "nominal — family size (GT3/LE3)",
    "Pstatus": "nominal — parent status (A/T)",
    "Mjob": "nominal — mother's job, no order",
    "Fjob": "nominal — father's job, no order",
    "reason": "nominal — reason for choosing school",
    "guardian": "nominal — student guardian",
    "schoolsup": "nominal — binary (yes/no)",
    "famsup": "nominal — binary (yes/no)",
    "paid": "nominal — binary (yes/no)",
    "activities": "nominal — binary (yes/no)",
    "nursery": "nominal — binary (yes/no)",
    "higher": "nominal — binary (yes/no)",
    "internet": "nominal — binary (yes/no)",
    "romantic": "nominal — binary (yes/no)",
}
STUDENT_REDUNDANT = []  # G1 and G2 are intermediate grades — relevant predictors
STUDENT_DESC = (
    "Portuguese student performance dataset. Predicts pass/fail from demographic, "
    "family, and school-related features. Most categorical columns are binary or low-cardinality "
    "nominal. Numeric columns (Medu, Fedu, traveltime, studytime, etc.) are ordinal integers "
    "on Likert-like scales (1-4 or 1-5) but already numeric. G1 and G2 are intermediate "
    "grades that are legitimate predictors of final_result."
)

# ========================================================================
# LIFE_EXPECTANCY dataset column semantics
# ========================================================================
LE_SEMANTICS = {
    "Country": "nominal — country name, high cardinality (183 unique)",
    "Status": "ordinal — development status (Developing < Developed)",
}
LE_REDUNDANT = []
LE_DESC = (
    "WHO Life Expectancy dataset. Regression task predicting life expectancy from "
    "health, economic, and demographic indicators across 183 countries. "
    "Country has very high cardinality (183 values). Status is binary ordinal."
)

# ========================================================================
# ML model configs
# ========================================================================
CLASSIFICATION_MODELS = {
    "logreg": "Logistic Regression (linear model)",
    "rf": "Random Forest (tree-based ensemble)",
    "knn": "K-Nearest Neighbors (distance-based)",
    "gbm": "Gradient Boosting Machine (tree-based ensemble)",
}
REGRESSION_MODELS = {
    "ridge": "Ridge Regression (linear model)",
    "rf": "Random Forest (tree-based ensemble)",
    "knn": "K-Nearest Neighbors (distance-based)",
    "gbm": "Gradient Boosting Machine (tree-based ensemble)",
}

DATASETS = {
    "adult": {
        "semantics": ADULT_SEMANTICS,
        "redundant": ADULT_REDUNDANT,
        "desc": ADULT_DESC,
        "models": CLASSIFICATION_MODELS,
        "task": "classification",
    },
    "diabetes": {
        "semantics": DIABETES_SEMANTICS,
        "redundant": DIABETES_REDUNDANT,
        "desc": DIABETES_DESC,
        "models": CLASSIFICATION_MODELS,
        "task": "classification",
    },
    "student": {
        "semantics": STUDENT_SEMANTICS,
        "redundant": STUDENT_REDUNDANT,
        "desc": STUDENT_DESC,
        "models": CLASSIFICATION_MODELS,
        "task": "classification",
    },
    "life_expectancy": {
        "semantics": LE_SEMANTICS,
        "redundant": LE_REDUNDANT,
        "desc": LE_DESC,
        "models": REGRESSION_MODELS,
        "task": "regression",
    },
}


def build_spec(ds_name, ds_info, model_key, model_desc):
    """Build a C4 context spec for one dataset × model combination."""
    is_tree = any(kw in model_desc.lower() for kw in ["tree", "forest", "gradient", "boosting"])

    # Build must_do based on model family
    must_do = ["handle_missing"]
    if is_tree:
        must_do.append("encode_categorical_per_column with ordinal as default for tree-based efficiency")
    else:
        must_do.append("encode_categorical_per_column: ordinal for ordered columns, one_hot for nominal")
        must_do.append("clip_outliers to protect distance/linear sensitivity")

    # Notes adapt to model type
    if is_tree:
        notes = (
            f"Tree-based model ({model_key}): ordinal encoding is efficient for all categoricals. "
            "Trees split on thresholds, so ordinal values work regardless of natural ordering. "
            "Outlier clipping is optional (trees are robust to outliers). "
            "Feature selection can reduce overfitting on wide datasets."
        )
    else:
        notes = (
            f"Linear/distance-based model ({model_key}): use one_hot for nominal categoricals to avoid "
            "false ordinal assumptions. Use ordinal ONLY for truly ordered columns (e.g. education levels). "
            "MUST clip outliers — linear and distance models are very sensitive to extreme values. "
            "Feature selection helps reduce multicollinearity."
        )

    return {
        ds_name: {
            "dataset_description": ds_info["desc"],
            "downstream_model": model_desc,
            "column_semantics": ds_info["semantics"],
            "redundant_features": ds_info["redundant"],
            "must_do": must_do,
            "must_not_do": [],
            "notes": notes,
        }
    }


# Generate all specs
for ds_name, ds_info in DATASETS.items():
    for model_key, model_desc in ds_info["models"].items():
        spec = build_spec(ds_name, ds_info, model_key, model_desc)
        out_path = OUT / f"{ds_name}_{model_key}.json"
        out_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] {out_path.name}")

print(f"\nGenerated {sum(len(d['models']) for d in DATASETS.values())} C4 context specs.")
