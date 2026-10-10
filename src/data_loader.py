"""Data loader module for Metro Interstate Traffic Volume dataset.

Provides robust dataset loading with path resolution, format validation,
and graceful error handling.
"""

from pathlib import Path
from typing import Optional, Union
import pandas as pd

# Expected columns in the UCI Metro Interstate Traffic Volume dataset
EXPECTED_COLUMNS = [
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    "date_time",
    "traffic_volume",
]

MANDATORY_COLUMNS = ["date_time", "traffic_volume"]


def resolve_dataset_path(provided_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the dataset file path with fallback to standard project locations.

    Parameters
    ----------
    provided_path : str or Path, optional
        User-specified path to the dataset file.

    Returns
    -------
    Path
        Resolved Path object pointing to the existing dataset.

    Raises
    ------
    FileNotFoundError
        If no dataset file can be located at the provided or candidate paths.
    """
    project_root = Path(__file__).resolve().parent.parent

    if provided_path is not None:
        p = Path(provided_path)
        candidates = []
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(Path.cwd() / p)
            candidates.append(project_root / p)
        for cand in candidates:
            if cand.exists() and cand.is_file():
                return cand.resolve()
        checked = "\n  - ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            f"User-specified dataset path '{provided_path}' was not found. Checked:\n  - {checked}"
        )

    # Standard default locations when no path is provided
    default_candidates = [
        project_root / "data" / "Metro_Interstate_Traffic_Volume.csv",
        project_root / "Metro_Interstate_Traffic_Volume.csv",
        Path.cwd() / "data" / "Metro_Interstate_Traffic_Volume.csv",
        Path.cwd() / "Metro_Interstate_Traffic_Volume.csv",
    ]

    for path in default_candidates:
        if path.exists() and path.is_file():
            return path.resolve()

    checked_locations = "\n  - ".join(str(p) for p in default_candidates)
    raise FileNotFoundError(
        f"Traffic dataset not found. Checked default locations:\n  - {checked_locations}\n"
        f"Please ensure 'Metro_Interstate_Traffic_Volume.csv' is placed inside "
        f"'{project_root / 'data'}'."
    )


def load_traffic_data(
    file_path: Optional[Union[str, Path]] = None,
    parse_dates: bool = True
) -> pd.DataFrame:
    """Load the Metro Interstate Traffic Volume dataset into a pandas DataFrame.

    Parameters
    ----------
    file_path : str or Path, optional
        Path to the CSV file. If None, default search paths are evaluated.
    parse_dates : bool, default True
        Whether to parse the 'date_time' column into datetime format.

    Returns
    -------
    pd.DataFrame
        Loaded dataset with intact records and parsed datetimes.

    Raises
    ------
    FileNotFoundError
        If the file cannot be located.
    ValueError
        If mandatory columns ('date_time', 'traffic_volume') are missing.
    RuntimeError
        If parsing or reading the CSV file encounters a malformed format.
    """
    resolved_path = resolve_dataset_path(file_path)

    try:
        df = pd.read_csv(resolved_path)
    except pd.errors.EmptyDataError as err:
        raise RuntimeError(
            f"Dataset at '{resolved_path}' is empty: {err}"
        ) from err
    except pd.errors.ParserError as err:
        raise RuntimeError(
            f"Failed to parse CSV at '{resolved_path}'. The file appears malformed: {err}"
        ) from err
    except UnicodeDecodeError as err:
        raise RuntimeError(
            f"Encoding error reading '{resolved_path}'. Expected standard UTF-8 CSV: {err}"
        ) from err
    except Exception as err:
        raise RuntimeError(
            f"Unexpected error loading dataset from '{resolved_path}': {err}"
        ) from err

    # Check for unexpected or missing mandatory columns
    missing_mandatory = [col for col in MANDATORY_COLUMNS if col not in df.columns]
    if missing_mandatory:
        raise ValueError(
            f"Dataset at '{resolved_path}' is missing mandatory column(s): {missing_mandatory}. "
            f"Available columns: {list(df.columns)}"
        )

    missing_expected = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_expected:
        print(f"[WARNING] Some standard columns are missing: {missing_expected}")

    unexpected_columns = [col for col in df.columns if col not in EXPECTED_COLUMNS]
    if unexpected_columns:
        print(f"[INFO] Additional columns found: {unexpected_columns}")

    # Parse date_time without dropping rows
    if parse_dates and "date_time" in df.columns:
        df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")

    return df
