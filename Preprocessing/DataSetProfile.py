"""Basic characteristics of a dataset."""
import sys

import pandas as pd

from text_preprocessing import TextPreprocessing


def load_dataset(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def profile(df):
    text_cols = TextPreprocessing.identify_text_columns(df)
    numeric_cols = [c for c in df.columns
                    if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    categorical_cols = [c for c in df.columns
                        if c not in text_cols and c not in numeric_cols
                        and (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
                             or isinstance(df[c].dtype, pd.CategoricalDtype)
                             or pd.api.types.is_bool_dtype(df[c]))]
    other_cols = [c for c in df.columns
                  if c not in text_cols + numeric_cols + categorical_cols]  # e.g. datetime

    missing = df.isna().sum()
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "numerical_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "text_columns": text_cols,
        "other_columns": other_cols,
        "missing_values": missing[missing > 0].to_dict(),
        "missing_total": int(missing.sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_counts": df.nunique().to_dict(),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 3),
    }


def print_report(p):
    print(f"Rows: {p['rows']}   Columns: {p['columns']}   Size in memory: {p['memory_mb']} MB")
    print(f"Numerical:   {p['numerical_columns']}")
    print(f"Categorical: {p['categorical_columns']}")
    print(f"Text:        {p['text_columns']}")
    if p["other_columns"]:
        print(f"Other (e.g. datetime): {p['other_columns']}")
    print(f"Missing values: {p['missing_total']} total  {p['missing_values'] or ''}")
    print(f"Duplicate rows: {p['duplicate_rows']}")
    print("Unique values per column:")
    for col, n in p["unique_counts"].items():
        print(f"    {col}: {n}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data = load_dataset(sys.argv[1])
    else:  # demo data
        data = pd.DataFrame({
            "id": [1, 2, 3, 4, 4],
            "price": [9.5, 12.0, None, 7.25, 7.25],
            "country": ["DE", "FR", "DE", "FR", "FR"],
            "review": ["great battery and screen", "slow shipping, bad support", "works fine for me",
                       "terrible customer service", "terrible customer service"],
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-04"]),
        })
    print_report(profile(data))