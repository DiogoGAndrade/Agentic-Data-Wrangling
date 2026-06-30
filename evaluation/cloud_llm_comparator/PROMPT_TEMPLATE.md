# Cloud LLM Comparator — Prompt Template

Use this EXACT prompt for ChatGPT, Gemini, Claude.ai, and Copilot.
Attach the relevant `context_<dataset>.json` file (or paste its contents).

---

## PROMPT (copy everything below this line)

You are an expert data scientist. I will give you a JSON file describing a tabular dataset.
Your task is to generate a data cleaning plan that will maximize the downstream ML model performance.

**Instructions:**
1. Read the dataset schema carefully (dtypes, missing values, cardinality).
2. Generate a JSON cleaning plan with key `"actions"` — a list of cleaning steps.
3. Each action must have: `"action"` (string), `"target_columns"` (list), `"params"` (dict).
4. Use ONLY these allowed actions (exact strings):

| Action | Purpose | Key params |
|--------|---------|-----------|
| `handle_missing` | Impute missing values | `{"strategy": "most_frequent"}` or `{"strategy": "median"}` |
| `encode_categorical_per_column` | Per-column encoding | `{"column_encodings": {"col": "ordinal"/"one_hot"}, "default_method": "one_hot"}` |
| `clip_outliers` | Clip numeric outliers | `{"method": "iqr", "iqr_k": 1.5}` |
| `transform_numeric_skewed` | Log/power transform for skewed columns | `{"skewness_threshold": 0.75, "method": "log1p", "model_family": "linear"}` |
| `add_missing_indicators` | Binary flag for high-NA columns | `{"min_na_rate": 0.05, "max_indicators": 10}` |
| `group_rare_categories` | Merge infrequent categories into "Other" | `{"min_frequency_pct": 0.01, "replacement_label": "Other"}` |
| `select_features` | Drop redundant/noisy columns | `{"drop_columns": ["col1", "col2"]}` |
| `bin_numeric` | Discretise continuous feature | `{"columns": ["col"], "n_bins": 5}` |

5. The downstream ML models tested are: Logistic Regression, Random Forest, KNN, Gradient Boosting.
6. Think about which columns have natural order (→ ordinal) vs nominal (→ one_hot).
7. For tree-based models (RF, GBM), ordinal encoding is preferred. For linear/distance models (LogReg, KNN), one_hot is preferred. Apply the most appropriate encoding per column.
8. Return ONLY valid JSON. No markdown, no explanation.

**Dataset context is attached / pasted below.**

---

## Expected output format (example — do NOT copy, generate your own):

```json
{
  "actions": [
    {
      "action": "handle_missing",
      "target_columns": ["workclass", "occupation"],
      "params": {"strategy": "most_frequent"}
    },
    {
      "action": "encode_categorical_per_column",
      "target_columns": [],
      "params": {
        "column_encodings": {
          "education": "ordinal",
          "occupation": "one_hot",
          "race": "one_hot"
        },
        "default_method": "one_hot"
      }
    }
  ]
}
```

---

## After getting the response

Save the JSON output to:
`evaluation/cloud_llm_comparator/plan_<llm>_<dataset>.json`

Examples:
- `plan_chatgpt_adult.json`
- `plan_gemini_heart.json`  
- `plan_claude_bank.json`
- `plan_copilot_diabetes.json`

Then run:
```powershell
python -m evaluation.apply_cloud_llm_plan --llm chatgpt --dataset adult
```
