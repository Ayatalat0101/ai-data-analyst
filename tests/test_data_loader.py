"""Tests for the data layer: valid file profile + every invalid-file case (T12)."""
from pathlib import Path

import pytest

from agent.data_loader import DataError, load_csv

SAMPLE = Path(__file__).parent.parent / "data" / "aid_distributions.csv"


@pytest.fixture(scope="module")
def loaded():
    return load_csv(SAMPLE.read_bytes(), SAMPLE.name)


def test_valid_file_shape(loaded):
    df, profile = loaded
    assert profile.rows == 60
    assert len(profile.columns) == 9


def test_column_kinds(loaded):
    _, p = loaded
    assert p.column("date").kind == "date"
    assert p.column("quantity").kind == "number"
    assert p.column("governorate").kind == "category"
    assert p.column("distribution_id").kind == "text"          # 60 unique ids


def test_missing_values_are_reported(loaded):
    _, p = loaded
    assert p.column("households_reached").missing == 2
    assert any("households_reached" in w for w in p.warnings)


def test_category_values_listed(loaded):
    _, p = loaded
    assert p.column("status").values == ["Cancelled", "Completed", "Pending"]


def test_schema_text_has_no_data_rows(loaded):
    """Privacy: the LLM gets the schema, never the records."""
    _, p = loaded
    schema = p.to_schema_text()
    assert "governorate" in schema
    assert "DST-" not in schema          # no record identifiers leak into the prompt


@pytest.mark.parametrize("raw, name, message", [
    (b"", "empty.csv", "empty"),
    (b"a,b\n", "header_only.csv", "no data rows"),
    (b"only_one_column\n1\n2\n", "one_col.csv", "at least 2 columns"),
    (b"id,value,Value\n1,2,3\n", "dups.csv", "Duplicate column names"),
    (b"id,,value\n1,2,3\n", "blank.csv", "no name"),
    (b"PK\x03\x04fake-excel-bytes", "renamed.csv", "Excel"),
    (b"a,b\n1,2\n", "data.xlsx", "not a .csv"),
])
def test_invalid_files_give_clear_errors(raw, name, message):
    with pytest.raises(DataError, match=message):
        load_csv(raw, name)


def test_arabic_windows_encoding_is_supported():
    raw = "المحافظة,العدد\nغزة,5\nرفح,7\n".encode("cp1256")
    df, p = load_csv(raw, "arabic.csv")
    assert list(df.columns) == ["المحافظة", "العدد"]
    assert df["العدد"].sum() == 12


def test_text_cells_are_trimmed():
    df, _ = load_csv(b"gov,qty\nRafah ,5\n Rafah,7\n", "spaces.csv")
    assert df["gov"].nunique() == 1
