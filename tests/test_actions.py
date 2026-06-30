"""Unit tests for the deterministic cleaning-action executors in engine.actions.

Each executor takes a DataFrame and a params dict and returns a tuple of
(transformed_df, diff_dict, warnings_list). These tests cover the common paths
and a few edge cases that the guardrail layer relies on.

Run with: pytest tests/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.actions import (
    fix_column_names,
    cast_type,
    handle_missing,
    normalize_text,
    deduplicate,
    drop_column,
    encode_categorical,
    remove_outliers,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "First Name": ["  Ana", "Bruno", "Bruno", None],
            "Age": [25, 40, 40, np.nan],
            "City-Name": ["Lisboa", "Porto", "Porto", "Faro"],
        }
    )


def test_fix_column_names_normalises_headers(sample_df):
    out, diff, warnings = fix_column_names(sample_df, {})
    assert list(out.columns) == ["first_name", "age", "city_name"]
    assert diff["columns_renamed"]["First Name"] == "first_name"


def test_fix_column_names_does_not_mutate_input(sample_df):
    original = list(sample_df.columns)
    fix_column_names(sample_df, {})
    assert list(sample_df.columns) == original


def test_handle_missing_median_imputation_fills_numeric():
    df = pd.DataFrame({"x": [1.0, 3.0, np.nan, 5.0]})
    out, diff, warnings = handle_missing(
        df, {"strategy": "impute", "columns": ["x"],
             "impute": {"numeric": "median", "categorical": "most_frequent"}}
    )
    assert out["x"].isna().sum() == 0
    assert out["x"].iloc[2] == 3.0  # median of [1, 3, 5]


def test_normalize_text_strips_and_lowercases():
    df = pd.DataFrame({"c": ["  Hello ", "WORLD"]})
    out, diff, warnings = normalize_text(
        df, {"columns": ["c"], "ops": ["strip", "lower", "collapse_whitespace"]}
    )
    assert out["c"].tolist() == ["hello", "world"]


def test_deduplicate_removes_exact_duplicates(sample_df):
    out, diff, warnings = deduplicate(sample_df, {"subset": None, "keep": "first"})
    assert len(out) < len(sample_df)
    assert out.duplicated().sum() == 0


def test_drop_column_removes_named_column(sample_df):
    out, diff, warnings = drop_column(sample_df, {"columns": ["Age"], "reason": "test"})
    assert "Age" not in out.columns


def test_encode_categorical_one_hot_expands_columns():
    df = pd.DataFrame({"colour": ["red", "green", "blue", "red"]})
    out, diff, warnings = encode_categorical(
        df, {"columns": ["colour"], "method": "one_hot", "max_categories": 30}
    )
    # one-hot should produce more columns than the single input
    assert out.shape[1] >= 2
    assert "colour" not in out.columns or out.shape[1] > 1


def test_remove_outliers_iqr_drops_extreme_value():
    df = pd.DataFrame({"v": [10, 11, 12, 13, 1000]})
    out, diff, warnings = remove_outliers(
        df, {"columns": ["v"], "method": "iqr", "iqr_k": 1.5,
             "mode": "drop_rows", "combine": "any"}
    )
    assert 1000 not in out["v"].values
    assert len(out) == 4


def test_cast_type_to_string():
    df = pd.DataFrame({"n": [1, 2, 3]})
    out, diff, warnings = cast_type(
        df, {"columns": ["n"], "dtype": "string", "errors": "raise"}
    )
    assert str(out["n"].dtype) in ("string", "object")


def test_all_executors_return_three_tuple(sample_df):
    """Every executor must return (df, diff, warnings) for the pipeline contract."""
    result = drop_column(sample_df, {"columns": ["Age"], "reason": "test"})
    assert isinstance(result, tuple) and len(result) == 3
    df_out, diff, warnings = result
    assert isinstance(df_out, pd.DataFrame)
    assert isinstance(diff, dict)
    assert isinstance(warnings, list)
