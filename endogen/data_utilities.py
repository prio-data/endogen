import os
import pandas as pd
import pyarrow.parquet as pq
from typing import Set
from pathlib import Path


def read_input_data(input_data: str | os.PathLike | pd.DataFrame) -> pd.DataFrame:
    """Reads input data to be used in estimation of statistical models and as initial values for simulation.

    Parameters
    ----------
    input_data : str | os.PathLike | pd.DataFrame
        Path to input data or a pandas.DataFrame object. Supports .csv and .parquet files.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    NotImplementedError
        Currently only supports .csv and .parquet files.
    ValueError
        Input data must be pandas.DataFrame if not path to .csv or .parquet file.
    """
    if isinstance(input_data, str) or isinstance(input_data, os.PathLike):
        input_data = Path(input_data)
        if input_data.suffix == ".csv":
            df = pd.read_csv(input_data)
        elif input_data.suffix == ".parquet":
            df = pq.read_table(input_data).to_pandas()
        else:
            raise NotImplementedError(
                "Currently only supports reading .csv and .parquet files."
            )
    elif isinstance(input_data, pd.DataFrame):
        df = input_data
    else:
        raise ValueError("Input data is not valid")

    if df.index.names != [None]:
        df.reset_index()

    return df


def compare_to_most_recent(
    df: pd.DataFrame,
    time_var: str,
    unit_var: str,
    alternative_time_comparison: int = None,
) -> tuple[list[tuple[int, Set[int]]], list[tuple[int, Set[int]]]]:
    """Compares all temporal cross-sections with the most recent, and finds the superfluous and missing units.
    Superfluous units are units found in cross-sections that is not the most recent, but is not found in the most recent cross-section.
    Missing units are units that are found in the most recent cross-section, but not found in other cross-sections.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe with panel data
    time_var : str
        A time-index variable of integer type (e.g., year).
    unit_var : str
        A unit-index variable of integer type (e.g., gwcode)
    alternative_time_comparison : int
        An alternative comparison period to use instead of the most recent. Must be an integer that is found in the time_var column.

    Returns
    -------
    tuple[list[tuple[int, Set[int]]], list[tuple[int, Set[int]]]]
        _description_
    """
    if df.index.names != [None]:
        df.reset_index()
    if alternative_time_comparison == None:
        most_recent_time = df[time_var].max()
    else:
        most_recent_time = alternative_time_comparison

    grouped = df.groupby(time_var)
    unit_sets = []
    for t, group in grouped:
        res = (t, set(group[unit_var].unique()))
        if t != most_recent_time:
            unit_sets.append(res)
        else:
            unit_sets.append(res)
            comparison_set = res
    missing = [(t, comparison_set[1] - s) for t, s in unit_sets]
    superfluous = [(t, s - comparison_set[1]) for t, s in unit_sets]
    return superfluous, missing


def generate_comparison_report(
    df: pd.DataFrame,
    time_var: str,
    unit_var: str,
    alternative_time_comparison: int = None,
) -> pd.DataFrame:
    """A report that is useful to understand how to build a complete and balanced dataset from a input panel data.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe with panel data
    time_var : str
        A time-index variable of integer type (e.g., year).
    unit_var : str
        A unit-index variable of integer type (e.g., gwcode)
    alternative_time_comparison : int
        An alternative comparison period to use instead of the most recent. Must be an integer that is found in the time_var column.

    Returns
    -------
    pd.DataFrame
        A dataframe with one observation per time-unit, indicating the superfluous and missing units across time.
    """
    grouped = df.groupby(time_var)
    superfluous, missing = compare_to_most_recent(
        df, time_var, unit_var, alternative_time_comparison
    )
    report = pd.DataFrame({"nobs": grouped.size()})
    report["superfluous"] = [s for _, s in superfluous]
    report["missing"] = [s for _, s in missing]
    return report


def drop_superfluous(
    df: pd.DataFrame,
    time_var: str,
    unit_var: str,
    alternative_time_comparison: int = None,
) -> pd.DataFrame:
    """Drop superfluous units from the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe with panel data
    time_var : str
        A time-index variable of integer type (e.g., year).
    unit_var : str
        A unit-index variable of integer type (e.g., gwcode)
    alternative_time_comparison : int
        An alternative comparison period to use instead of the most recent. Must be an integer that is found in the time_var column.

    Returns
    -------
    pd.DataFrame
        A dataframe where units of observation that is not in the lastest time-period is dropped.
    """
    report = generate_comparison_report(
        df, time_var, unit_var, alternative_time_comparison
    )
    report = report.reset_index()
    df = pd.merge(df, report[[time_var, "superfluous"]], on=time_var)
    df["superfluous"] = df.apply(lambda x: x[unit_var] in x["superfluous"], axis=1)
    df = df[~df.superfluous]
    return df.drop(columns="superfluous")


def drop_missing_units(
    df: pd.DataFrame,
    time_var: str,
    unit_var: str,
    alternative_time_comparison: int = None,
):
    """Drop missing units from the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe with panel data
    time_var : str
        A time-index variable of integer type (e.g., year).
    unit_var : str
        A unit-index variable of integer type (e.g., gwcode)
    alternative_time_comparison : int
        An alternative comparison period to use instead of the most recent. Must be an integer that is found in the time_var column.

    Returns
    -------
    pd.DataFrame
        A dataframe where years with missing units of observation compared to the latest time-period is dropped.
    """
    report = generate_comparison_report(
        df, time_var, unit_var, alternative_time_comparison
    )
    earliest_time_without_any_units_missing = report.missing.apply(
        lambda x: len(x) == 0
    ).idxmax()
    return df.loc[df[time_var] >= earliest_time_without_any_units_missing]
