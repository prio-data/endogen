from .utilities import PanelUnits

from mlforecast.forecast import MLForecast
from mlforecast.utils import PredictionIntervals

from typing import Literal

import numpy as np
import pandas as pd
import itertools
import xarray


def percentile_hi_lo(interval: float, type: Literal["lo", "hi"]) -> float:
    """Calculate lo/hi percentile from predictive interval

    Parameters
    ----------
    interval : float
        The predictive interval in percent.
    type : Literal["lo", "hi"]
        Whether to return the low or high percentile from the interval.

    Returns
    -------
    float
        A percentile from 0 - 1.
    """
    alpha = (100 - interval) / 100
    if type == "lo":
        return alpha / 2
    if type == "hi":
        return 1 - (alpha / 2)


def setup_mlforecast_bins(
    model: MLForecast, levels=list[float]
) -> tuple[dict[str, str], list[float]]:
    """A helper function to rename columns from MLForecast models with PredictiveIntervals. These are named in terms of the prediction interval and
    a lo/hi indicator. This function converts this to percentiles and returns a dictionary for easy renaming in Pandas.

    The idea here is to create an equally binned histogram over the percentiles in a prediction distribution, finding the prediction at the middle of each bin.

    Parameters
    ----------
    model : MLForecast
        A `mlforecast.forecast.MLForecast` model object
    levels : list[float]
        A list of prediction intervals. E.g., [50, 90] gives the 50% and 90% prediction interval

    Returns
    -------
    tuple[dict[str, str], list[float]]
        A tuple with a dictionary with the renaming scheme for use in Pandas and the list of percentiles.
    """

    var_names = {
        f"{m}-{t}-{l}": f"{m}-{percentile_hi_lo(l, t)}"
        for m, t, l in list(
            itertools.product(model.models.keys(), ["lo", "hi"], levels)
        )
    }

    # this is probably not generalizable to any type if integer list. idea is to eavenly spread equally spaced bins across 0-1 and get the percentile for the middle of the bin
    p = [1 / len(var_names)] * len(var_names)
    return var_names, p


def forecast_mlforecast(
    t: int,
    s: int,
    model: MLForecast,
    xdata: xarray.Dataset,
    pnames: PanelUnits,
    output_var: str,
    input_vars: list[str],
    levels=list[float],
) -> pd.DataFrame:
    """A prediction function adapter for MLForecast fitted with PredictionIntervals drawing predictions from the (approx./stepwise)
    full predictive distribution.

    Parameters
    ----------
    t : int
        Time index to forecast
    s : int
        Simulation index to forecast
    model : MLForecast
        The MLForecast object
    xdata : xarray.Dataset
        The input data used in forecasting
    pnames : PanelUnits
        The internal index naming convention used.
    output_var : str
        The output variable
    input_vars : list[str]
        The list of input variables
    levels : list[float]
        The list of prediction intervals that the MLForecast model has been fitted with.

    Returns
    -------
    pd.DataFrame
        A properly indexed pandas.DataFrame with a single draw from the full predictive distribution for all units at time t.
    """
    df = (
        xdata[list(itertools.chain([output_var], input_vars))]
        .to_dataframe()
        .loc[t, :, s]
    )
    df = df.rename(columns={output_var: "y"})
    df.reset_index(inplace=True)
    df["ds"] = t

    X = xdata[input_vars].to_dataframe().loc[t, :, s]
    X.reset_index(inplace=True)
    X["ds"] = t + 1

    res = model.predict(h=1, level=levels, new_df=df, X_df=X)
    var_names, p = setup_mlforecast_bins(model, levels)
    res = res.rename(columns=var_names)
    res.set_index(pnames.internal_index, inplace=True)
    res = res.drop(columns=model.models.keys())
    res = res.reindex(sorted(res.columns), axis=1)
    res = res.apply(
        lambda x: np.random.choice(x.tolist(), size=1, p=p),
        axis=1,
    ).explode()
    return res
