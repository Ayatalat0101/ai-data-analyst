"""
Data layer: load a CSV safely and describe it.

Why this file exists
--------------------
The agent can only reason about data it understands. Before any question is
asked we (1) reject files that would crash or mislead the app and (2) build a
"profile" (the schema) that the agent receives instead of the raw rows.

No Streamlit import here: this module is pure Python so it can be unit-tested.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

import pandas as pd

MAX_BYTES = 10 * 1024 * 1024        # 10 MB, from the input contract
MAX_CATEGORY_VALUES = 25            # a text column with <= 25 distinct values is a "category"
ENCODINGS = ("utf-8-sig", "utf-8", "cp1256")   # cp1256 = Arabic Windows / Excel exports


class DataError(Exception):
    """A problem the USER can fix. The message is shown in the UI as-is."""


@dataclass
class ColumnInfo:
    name: str
    kind: str                     # "number" | "category" | "text" | "date"
    missing: int
    unique: int
    examples: list[str]
    values: list[str] | None = None   # all distinct values, only for categories
    min: float | str | None = None
    max: float | str | None = None


@dataclass
class DatasetProfile:
    file_name: str
    rows: int
    columns: list[ColumnInfo]
    duplicate_rows: int
    warnings: list[str] = field(default_factory=list)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def column(self, name: str) -> ColumnInfo | None:
        return next((c for c in self.columns if c.name == name), None)

    def columns_of_kind(self, *kinds: str) -> list[str]:
        return [c.name for c in self.columns if c.kind in kinds]

    def to_schema_text(self) -> str:
        """Compact schema for the LLM prompt: names, types, allowed values.
        Deliberately contains NO data rows (privacy decision in PROJECT_PLAN §11)."""
        lines = [f"Dataset '{self.file_name}': {self.rows} rows"]
        for c in self.columns:
            line = f"- {c.name} ({c.kind}"
            if c.missing:
                line += f", {c.missing} missing"
            line += ")"
            if c.values is not None:
                line += f" values: {c.values}"
            elif c.kind in ("number", "date"):
                line += f" range: {c.min} to {c.max}"
            lines.append(line)
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def _decode(raw: bytes) -> str:
    if raw[:4] == b"PK\x03\x04":
        raise DataError("This looks like an Excel (.xlsx) or zip file renamed to .csv. "
                        "In Excel use File → Save As → CSV UTF-8, then upload again.")
    if b"\x00" in raw[:2048]:
        raise DataError("This file is binary, not a text CSV.")
    for enc in ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise DataError("Could not read the text encoding. Save the file as 'CSV UTF-8'.")


def _check_header(text: str) -> list[str]:
    """Read the header row ourselves: pandas silently renames duplicates (a, a.1)
    and blanks (Unnamed: 3), which would hide the real problem from the user."""
    try:
        header = next(csv.reader(io.StringIO(text)))
    except StopIteration:
        raise DataError("The file is empty.")
    names = [h.strip() for h in header]
    if len(names) < 2:
        raise DataError("The file needs at least 2 columns. Check that values are separated by commas.")
    blanks = [i + 1 for i, n in enumerate(names) if not n]
    if blanks:
        raise DataError(f"Column(s) at position {blanks} have no name. Add a header for every column.")
    seen, dups = set(), []
    for n in names:
        if n.lower() in seen:
            dups.append(n)
        seen.add(n.lower())
    if dups:
        raise DataError(f"Duplicate column names: {sorted(set(dups))}. Every column needs a unique name.")
    return names


def _detect_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert text columns that are clearly dates (ISO-like, >= 90% parseable)."""
    for col in df.select_dtypes(include=["object", "string"]).columns:
        sample = df[col].dropna().astype(str)
        if sample.empty or not sample.str.match(r"^\d{4}-\d{1,2}-\d{1,2}").mean() >= 0.9:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
        if parsed.notna().sum() >= 0.9 * df[col].notna().sum():
            df[col] = parsed
    return df


def load_csv(raw: bytes, file_name: str) -> tuple[pd.DataFrame, DatasetProfile]:
    """Validate + load. Raises DataError with a user-friendly message on any problem."""
    if not file_name.lower().endswith(".csv"):
        raise DataError(f"'{file_name}' is not a .csv file.")
    if len(raw) == 0:
        raise DataError("The file is empty.")
    if len(raw) > MAX_BYTES:
        raise DataError(f"The file is {len(raw) / 1e6:.1f} MB. The limit is {MAX_BYTES // 1_000_000} MB.")

    text = _decode(raw)
    names = _check_header(text)

    try:
        df = pd.read_csv(io.StringIO(text), skipinitialspace=True)
    except pd.errors.ParserError as err:
        raise DataError(f"Rows have an inconsistent number of columns: {err}")

    df.columns = names
    if df.empty:
        raise DataError("The file has a header but no data rows.")

    # strip spaces inside text cells ("Rafah " -> "Rafah")
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()

    df = _detect_dates(df)
    return df, profile_dataframe(df, file_name)


# ----------------------------------------------------------------------------
# Profiling
# ----------------------------------------------------------------------------
def _kind(s: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return "number"
    return "category" if s.nunique(dropna=True) <= MAX_CATEGORY_VALUES else "text"


def _fmt(v) -> str:
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat()
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def profile_dataframe(df: pd.DataFrame, file_name: str) -> DatasetProfile:
    cols, warnings = [], []
    for name in df.columns:
        s = df[name]
        kind = _kind(s)
        non_null = s.dropna()
        info = ColumnInfo(
            name=name, kind=kind, missing=int(s.isna().sum()),
            unique=int(s.nunique(dropna=True)),
            examples=[_fmt(v) for v in non_null.unique()[:3]],
        )
        if kind == "category":
            info.values = sorted(_fmt(v) for v in non_null.unique())
        if kind in ("number", "date") and not non_null.empty:
            info.min, info.max = _fmt(non_null.min()), _fmt(non_null.max())
        if info.missing:
            warnings.append(f"'{name}' has {info.missing} missing value(s). "
                            "They are excluded from calculations on that column and reported.")
        if non_null.empty:
            warnings.append(f"'{name}' is completely empty.")
        cols.append(info)

    dup = int(df.duplicated().sum())
    if dup:
        warnings.append(f"{dup} fully duplicated row(s) found. They are kept; check the source file.")
    return DatasetProfile(file_name=file_name, rows=len(df), columns=cols,
                          duplicate_rows=dup, warnings=warnings)
