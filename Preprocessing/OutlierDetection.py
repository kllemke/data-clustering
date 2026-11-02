"""Identify potential outliers in numerical columns."""
"""Constant columns and non-numeric columns are skipped. If there are no usable numerical columns, the report says the check isn’t possible.
The distance method needs at least two numerical columns.
IQR and z-score can disagree, as they did on the demo’s weight column. IQR is stricter on skewed data, while z-scores are easily inflated by the outliers themselves.
My first version flagged 9 of 100 rows with the distance method, which is far too many. I tightened it to a z-score threshold of 3 on the distances, which gave the single flagged row in the demo.
Flagged values are candidates to inspect, not errors. A large value can be perfectly valid, so the code never removes anything on its own
"""

import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def numeric_columns(df):
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def iqr_outliers(s, factor=1.5):
    """Boolean mask: value lies outside [Q1 - factor*IQR, Q3 + factor*IQR]."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(False, index=s.index)
    return (s < q1 - factor * iqr) | (s > q3 + factor * iqr)


def zscore_outliers(s, threshold=3.0):
    """Boolean mask: |z-score| above threshold."""
    if s.std() == 0:
        return pd.Series(False, index=s.index)
    return ((s - s.mean()) / s.std()).abs() > threshold


def distance_outliers(df, columns, k=5, threshold=3.0):
    """Preliminary multivariate check: mean distance to the k nearest neighbours
    (on standardized data). Rows whose distance has a z-score above the
    threshold are flagged. Rows with missing values are skipped."""
    data = df[columns].dropna()
    k = min(k, len(data) - 1)
    if len(columns) < 1 or k < 1:
        return pd.Series(dtype=bool)
    X = StandardScaler().fit_transform(data)
    dist, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    score = pd.Series(dist[:, 1:].mean(axis=1), index=data.index)  # skip self
    return zscore_outliers(score, threshold)


def detect_outliers(df):
    cols = [c for c in numeric_columns(df) if df[c].nunique() > 1]
    per_column = {}
    for c in cols:
        s = df[c].dropna()
        per_column[c] = {"iqr": int(iqr_outliers(s).sum()),
                         "zscore": int(zscore_outliers(s).sum())}
    multi = distance_outliers(df, cols) if len(cols) >= 2 else pd.Series(dtype=bool)
    distance_rows = list(multi[multi].index)

    has_outliers = any(v["iqr"] or v["zscore"] for v in per_column.values()) or bool(distance_rows)
    return {"columns_checked": cols, "per_column": per_column,
            "distance_outlier_rows": distance_rows, "has_potential_outliers": has_outliers}


def print_report(r):
    if not r["columns_checked"]:
        print("No usable numerical columns -> outlier check not possible.")
        return
    print("Potential outliers found." if r["has_potential_outliers"] else "No potential outliers found.")
    print(f"Columns checked: {r['columns_checked']}")
    print("Per column (number of flagged values):")
    for c, v in r["per_column"].items():
        print(f"    {c}: IQR={v['iqr']}  z-score={v['zscore']}")
    rows = r["distance_outlier_rows"]
    print(f"Distance-based (multivariate) flagged rows: {rows if rows else 'none'}")
    print("Note: flagged values are candidates to inspect, not errors to delete automatically.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        data = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    else:  # demo data
        rng = np.random.default_rng(0)
        data = pd.DataFrame({"price": rng.normal(50, 5, 100), "weight": rng.normal(10, 1, 100)})
        data.loc[7, "price"] = 400          # extreme single value
    print_report(detect_outliers(data))