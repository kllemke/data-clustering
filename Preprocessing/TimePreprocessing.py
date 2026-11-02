"""Detect and prepare time-related data in a DataFrame."""
import warnings

import pandas as pd


def _to_datetime(s):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(s, errors="coerce")


class TimePreprocessing:
    def __init__(self, parse_threshold=0.9):
        self.parse_threshold = parse_threshold  # share of values that must parse as dates

    # ---------- 1. identify timestamp columns ----------
    def identify_timestamp_columns(self, df):
        """Datetime columns, plus text columns where most values parse as dates.
        Numeric columns are ignored (a number alone is not clearly a time)."""
        found = []
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_datetime64_any_dtype(s):
                found.append(col)
            elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
                s = s.dropna().astype(str).str.strip()
                s = s[s != ""]
                if s.empty or pd.to_numeric(s, errors="coerce").notna().mean() > 0.5:
                    continue  # empty or numeric-like text
                if _to_datetime(s).notna().mean() >= self.parse_threshold:
                    found.append(col)
        return found

    # ---------- 2. sort chronologically ----------
    @staticmethod
    def sort_chronologically(df, column):
        """Return a copy with `column` parsed as datetime and sorted ascending.
        Rows with unparseable/missing timestamps are dropped."""
        out = df.copy()
        out[column] = _to_datetime(out[column])
        return out.dropna(subset=[column]).sort_values(column).reset_index(drop=True)

    # ---------- 3. extract features ----------
    @staticmethod
    def extract_features(df, column):
        """Return a copy with calendar features added as `<column>_<feature>`."""
        out = df.copy()
        t = _to_datetime(out[column])
        out[f"{column}_year"] = t.dt.year
        out[f"{column}_quarter"] = t.dt.quarter
        out[f"{column}_month"] = t.dt.month
        out[f"{column}_day"] = t.dt.day
        out[f"{column}_dayofweek"] = t.dt.dayofweek   # Monday=0
        out[f"{column}_dayofyear"] = t.dt.dayofyear
        out[f"{column}_is_weekend"] = (t.dt.dayofweek >= 5).astype(int)
        if (t.dt.hour != 0).any():                      # only if times are present
            out[f"{column}_hour"] = t.dt.hour
        return out

    # ---------- 4. time series check ----------
    def check_time_series(self, df, column):
        """Heuristic. Likely a time series if timestamps are (almost) unique
        AND either already in order or regularly spaced."""
        t = _to_datetime(df[column]).dropna()
        if len(t) < 3:
            return {"column": column, "likely_time_series": False,
                    "reasons": ["fewer than 3 valid timestamps"]}

        unique_ratio = t.nunique() / len(t)
        in_order = bool(t.is_monotonic_increasing or t.is_monotonic_decreasing)
        diffs = t.sort_values().drop_duplicates().diff().dropna()
        regular = float((diffs == diffs.median()).mean()) if len(diffs) else 0.0
        try:
            freq = pd.infer_freq(t.sort_values().drop_duplicates())
        except (ValueError, TypeError):
            freq = None

        likely = unique_ratio >= 0.9 and (in_order or regular >= 0.8)
        reasons = [
            f"{unique_ratio:.0%} of timestamps are unique",
            "rows are already in chronological order" if in_order else "rows are not in chronological order",
            f"{regular:.0%} of gaps equal the median gap ({diffs.median()})",
        ]
        if freq:
            reasons.append(f"inferred frequency: {freq}")
        return {"column": column, "likely_time_series": likely, "reasons": reasons}

    def analyze(self, df):
        """Run detection on all candidate columns and print a clear verdict."""
        cols = self.identify_timestamp_columns(df)
        if not cols:
            print("No timestamp columns found -> NOT a time series.")
            return []
        results = [self.check_time_series(df, c) for c in cols]
        for r in results:
            verdict = "LIKELY a time series" if r["likely_time_series"] else "NOT likely a time series"
            print(f"[{r['column']}] {verdict}")
            for reason in r["reasons"]:
                print(f"    - {reason}")
        return results


if __name__ == "__main__":
    #Dummy Data
    ts = pd.DataFrame({
        "date": ["2026-01-03", "2026-01-01", "2026-01-02", "2026-01-04", "2026-01-05"],
        "sales": [30, 10, 20, 40, 50],
        "note": ["a b", "c d", "e f", "g h", "i j"],
    })
    people = pd.DataFrame({
        "birthday": ["1990-05-01", "1985-03-12", "1990-05-01", "2001-11-30", "1985-03-12"],
        "name": ["Ann", "Bob", "Cy", "Di", "Ed"],
    })

    prep = TimePreprocessing()
    prep.analyze(ts)
    print()
    prep.analyze(people)

    sorted_ts = prep.sort_chronologically(ts, "date")
    print()
    print(prep.extract_features(sorted_ts, "date").drop(columns="note"))