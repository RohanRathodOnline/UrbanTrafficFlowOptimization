"""Inspection module for Metro Interstate Traffic Volume dataset.

Analyzes dimensions, data types, missing records, duplicates, date coverage,
and statistical characteristics of the traffic volume target variable.
"""

from typing import Any, Dict
import pandas as pd


def inspect_traffic_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform comprehensive data inspection on the traffic dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded dataset with parsed datetimes.

    Returns
    -------
    dict
        Structured dictionary containing all computed inspection metrics.
    """
    rows, cols = df.shape

    # 1. First 5 rows
    head_rows = df.head(5)

    # 2. Column names and types
    col_dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # 3. Missing values per column
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / rows) * 100.0

    # 4. Duplicate rows
    duplicate_count = int(df.duplicated().sum())

    # 5. Invalid dates (NaT in parsed date_time column)
    invalid_dates_count = 0
    date_range_str = "N/A"
    if "date_time" in df.columns:
        invalid_dates_count = int(df["date_time"].isna().sum())
        valid_dates = df["date_time"].dropna()
        if not valid_dates.empty:
            start_date = valid_dates.min()
            end_date = valid_dates.max()
            date_range_str = f"{start_date} to {end_date}"

    # 6. Target statistics (traffic_volume)
    target_stats: Dict[str, Any] = {}
    missing_target_count = 0
    if "traffic_volume" in df.columns:
        missing_target_count = int(df["traffic_volume"].isna().sum())
        tv_series = df["traffic_volume"].dropna()
        if not tv_series.empty:
            target_stats = {
                "count": int(tv_series.count()),
                "mean": float(tv_series.mean()),
                "std": float(tv_series.std()),
                "min": float(tv_series.min()),
                "25%": float(tv_series.quantile(0.25)),
                "50% (median)": float(tv_series.median()),
                "75%": float(tv_series.quantile(0.75)),
                "max": float(tv_series.max()),
                "skewness": float(tv_series.skew()),
            }

    return {
        "rows": rows,
        "columns": cols,
        "head": head_rows,
        "column_dtypes": col_dtypes,
        "missing_counts": missing_counts.to_dict(),
        "missing_percentages": missing_percentages.to_dict(),
        "duplicate_rows": duplicate_count,
        "invalid_dates_count": invalid_dates_count,
        "missing_target_count": missing_target_count,
        "date_range": date_range_str,
        "target_stats": target_stats,
    }


def print_inspection_report(results: Dict[str, Any], file_path: str = "") -> None:
    """Print a clean, structured inspection report to stdout.

    Parameters
    ----------
    results : dict
        Inspection metrics returned by `inspect_traffic_dataset`.
    file_path : str, optional
        Path to the inspected dataset file for context.
    """
    sep_heavy = "=" * 70
    sep_light = "-" * 70

    print(sep_heavy)
    print(" URBAN TRAFFIC FLOW OPTIMIZATION - INITIAL DATASET INSPECTION")
    print(sep_heavy)
    if file_path:
        print(f"Dataset Path        : {file_path}")
    print(f"Total Rows          : {results['rows']:,}")
    print(f"Total Columns       : {results['columns']}")
    print(f"Date Coverage Range : {results['date_range']}")
    print(f"Duplicate Rows      : {results['duplicate_rows']:,}")
    print(f"Invalid Dates (NaT) : {results['invalid_dates_count']}")
    print(f"Missing Target (NA) : {results['missing_target_count']}")
    print(sep_light)

    print("\n[1] FIRST 5 ROWS:")
    print(results["head"].to_string())

    print("\n" + sep_light)
    print("[2] COLUMN NAMES, DATA TYPES & MISSING VALUES:")
    col_summary = []
    for col, dtype in results["column_dtypes"].items():
        n_missing = results["missing_counts"].get(col, 0)
        pct_missing = results["missing_percentages"].get(col, 0.0)
        col_summary.append({
            "Column": col,
            "Type": dtype,
            "Missing (Count)": n_missing,
            "Missing (%)": f"{pct_missing:.2f}%",
        })
    summary_df = pd.DataFrame(col_summary)
    print(summary_df.to_string(index=False))

    print("\n" + sep_light)
    print("[3] DESCRIPTIVE STATISTICS FOR TARGET ('traffic_volume'):")
    if results["target_stats"]:
        for metric, val in results["target_stats"].items():
            if isinstance(val, int) or metric == "count":
                print(f"  {metric:<15}: {int(val):,}")
            else:
                print(f"  {metric:<15}: {val:,.2f}")
    else:
        print("  [ERROR] 'traffic_volume' column not found or contains no valid values.")

    print(sep_heavy)
    print(" VERIFICATION SUMMARY:")
    expected_rows = 48204
    expected_cols = 9
    row_diff = results["rows"] - expected_rows
    if row_diff == 0:
        print(f"  [OK] Row count matches expected UCI benchmark: {results['rows']:,} rows.")
    else:
        print(f"  [NOTE] Row count is {results['rows']:,} (benchmark: {expected_rows:,}, diff: {row_diff:+d}).")

    if results["columns"] == expected_cols:
        print(f"  [OK] Column count matches expected UCI benchmark: {results['columns']} columns.")
    else:
        print(f"  [NOTE] Column count is {results['columns']} (expected: {expected_cols}).")

    if results["invalid_dates_count"] == 0 and results["missing_target_count"] == 0:
        print("  [OK] Integrity verified: 0 invalid timestamps and 0 missing target values.")
    else:
        print("  [WARNING] Dataset contains missing dates or targets requiring future preprocessing.")
    print(sep_heavy)
