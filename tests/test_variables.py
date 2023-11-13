import pytest
from endogen.variables import (
    Variable,
    VariableDifference,
    VariableLag,
    VariableRolling,
    VariableSingleEdge,
    VariableTransform,
)
import pandas as pd
import xarray
import numpy as np


@pytest.fixture
def my_df() -> pd.DataFrame:
    d = {"v1": [1, 2, 4, 6, 3, 1.5], "v2": [3, 4, 5, 7, 5, 3]}
    idx = pd.MultiIndex.from_arrays(
        [[1, 2, 3, 1, 2, 3], [10, 10, 10, 20, 20, 20]], names=("ds", "unique_id")
    )
    return pd.DataFrame(data=d, index=idx)


@pytest.fixture
def my_xarray(my_df) -> xarray.Dataset:
    return my_df.to_xarray()


@pytest.fixture
def diff_var() -> VariableDifference:
    return VariableDifference(output_var="out", subset=0, type="growth", input_var="v1")


@pytest.fixture
def lag_var() -> VariableLag:
    return VariableLag(output_var="out", subset=0, input_var="v2", num_lag=1)


@pytest.fixture
def rolling_var() -> VariableRolling:
    return VariableRolling(
        output_var="out",
        subset=0,
        input_var="v2",
        window=2,
        fun="sum",
        window_type="normal",
    )


def test_variable_node():
    var = Variable(output_var="out", subset=0)
    assert var.node == ("out", {"model": var, "subset": 0})


def test_difference_calc(my_xarray: xarray.Dataset, diff_var: VariableDifference):
    expected = np.array([np.nan, 1, 1])
    my_out = diff_var.calc(xd=my_xarray).sel(unique_id=10)
    assert np.array_equal(my_out, expected, equal_nan=True)
    assert my_out.name == "out"


def test_difference_calc_compared_to_pandas(
    my_xarray: xarray.Dataset, diff_var: VariableDifference
):
    expected = (
        my_xarray.to_dataframe().groupby(level=[1]).pct_change().to_xarray()["v1"]
    )
    my_out = diff_var.calc(xd=my_xarray)
    assert np.array_equal(my_out, expected, equal_nan=True)
    assert my_out.name == "out"


def test_lag_calc(my_xarray: xarray.Dataset, lag_var: VariableLag):
    expected = np.array([np.nan, 7, 5])
    my_out = lag_var.calc(xd=my_xarray).sel(unique_id=20)
    assert np.array_equal(my_out, expected, equal_nan=True)
    assert my_out.name == "out"


def test_rolling_calc(my_xarray: xarray.Dataset, rolling_var: VariableRolling):
    expected = np.array([np.nan, 7, 9])
    my_out = rolling_var.calc(xd=my_xarray).sel(unique_id=10)
    assert np.array_equal(my_out, expected, equal_nan=True)
    assert my_out.name == "out"


def test_rolling_wrong_fun(my_xarray: xarray.Dataset):
    with pytest.raises(NotImplementedError):
        VariableRolling("out", 0, "in", 2, "exp", "normal")


def test_rolling_wrong_window_type(my_xarray: xarray.Dataset):
    with pytest.raises(NotImplementedError):
        VariableRolling("out", 0, "in", 2, "mean", "half-life")


def test_difference_wrong_type(my_xarray: xarray.Dataset):
    with pytest.raises(NotImplementedError):
        VariableDifference("out", 0, "in", "grwth")
