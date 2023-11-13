from dataclasses import dataclass, field
from typing import List, Literal, Any, Tuple, Mapping
import xarray
from xarray.core.rolling import DataArrayRolling
from xarray.core.rolling_exp import RollingExp
import pandas as pd
from formulae import design_matrices
import numpy as np  # to be used in design_matrices
import scipy  # to be used in design_matrices


@dataclass(eq=True, frozen=True)
class Variable:
    """A variable class that holds the information necessary to represent a variable model or transformation in the simulation system.
    Not used for statistical models (see `config.InputModel`).

    Parameters
    ----------
    output_var : str
        Name of the output variable
    subset : int
        Endogenous simulation requires that all variables are fully specified in a circular fashion.
        At the same time, there cannot be any circular definitions in the transformation step, nor in the forecast step.
        If the variable is subset == 0, it is estimated/calculated in the transformation step, if it is 1, it is in the forecast step.

    Raises
    ------
    NotImplementedError
        This is just the base class that other Variable classes inherits. This class is not meant to be instantiated.
    """

    output_var: str
    subset: int

    @property
    def node(self) -> Tuple[str, Mapping[str, Any]]:
        """A node representation that interface well with NetworkX graphs.

        Returns
        -------
        Tuple[str, Mapping[str, Any]]
            A tuple where the first element is the output variable name ("node"), and the second element is a dictionary of node data.
        """
        return (self.output_var, {"model": self, "subset": self.subset})

    def calc(self, xd: xarray.Dataset) -> xarray.DataArray:
        """A function that takes input data that includes input-variables required to calculate the output_var.

        Parameters
        ----------
        xd : xarray.Dataset
            A xarray.Dataset, often `EndogenousSystem._xa` (simulation data), or the `EndogenousSystem._past`.

        Returns
        -------
        xarray.DataArray
            Returns a DataArray with the output_var, properly indexed.

        Raises
        ------
        NotImplementedError
            This is just the base class that other Variable classes inherits. This class is not meant to be instantiated.
        """
        raise NotImplementedError


@dataclass(eq=True, frozen=True)
class VariableSingleEdge(Variable):
    """A variable class that holds the information necessary to represent a variable model or transformation in the simulation system.
    Not used for statistical models (see `config.InputModel`). Helper class that I will probably regret.
    Used for all Variables that only take one input variable.

    Parameters
    ----------
    output_var : str
        Name of the output variable
    subset : int
        Endogenous simulation requires that all variables are fully specified in a circular fashion.
        At the same time, there cannot be any circular definitions in the transformation step, nor in the forecast step.
        If the variable is subset == 0, it is estimated/calculated in the transformation step, if it is 1, it is in the forecast step.
    input_var : str
        Name of the input variable
    """

    input_var: str

    @property
    def edges(self) -> List[Tuple[str, str]]:
        """An edge representation (list of tuples) that interface well with NetworkX graphs.

        Returns
        -------
        List[Tuple[str, str]]
            A graph edge describing the link between the input_var and the output_var.
        """
        return [(self.input_var, self.output_var)]


@dataclass(eq=True, frozen=True)
class VariableTransform(Variable):
    """A class holding the information necessary to do a mathematical transformation of input-variables to calculate a new output variable.
    This class only supports transformations that are static in time, e.g., log-transformations, multiplications, etc. It is using `formulae.design_matrices()`
    to do the transformation.

    Caution! The Wilkinson formula language have some gotcha's. For instance, "var1 + var2" will give you a matrix with two columns, not a matrix
    with one column being the sum of var1 and var2. To achieve the latter, you need to write "I(var1 + var2)". Any numpy ("np.") and scipy ("scipy.")
    tranformative function is in principle also supported. E.g., "np.sum(var1, var2)" would achieve the same.
    It is also important to note the difference between "var1:var2" and "var1*var2".

    Warning! Currently, there is nothing stopping you making a design matrix with multiple columns. This is not supported.

    Parameters
    ----------
    output_var : str
        Name of the output variable
    subset : int
        Endogenous simulation requires that all variables are fully specified in a circular fashion.
        At the same time, there cannot be any circular definitions in the transformation step, nor in the forecast step.
        If the variable is subset == 0, it is estimated/calculated in the transformation step, if it is 1, it is in the forecast step.
    formula : str
        A Wilkinson formula supported by `formulae`. See https://bambinos.github.io/formulae/notebooks/getting_started.html#User-guide.
    input_vars : list[str]
        List with names of the input variables.
    """

    formula: str
    input_vars: List[str] = field(default_factory=list)

    @property
    def edges(self) -> List[Tuple[str, str]]:
        """The edges between input variables and the transformed output variable that interface well with NetworkX graphs.

        Returns
        -------
        List[Tuple[str, str]]
            A list of graph edges describing the link between the input_var and the output_var.
        """
        return [(input_var, self.output_var) for input_var in self.input_vars]

    def calc(self, xd: xarray.Dataset) -> xarray.DataArray:
        """A variable transformation function that takes input data that includes input-variables required to calculate the output_var.

        Parameters
        ----------
        xd : xarray.Dataset
            A xarray.Dataset, often `EndogenousSystem._xa` (simulation data), or the `EndogenousSystem._past`.

        Returns
        -------
        xarray.DataArray
            Returns a DataArray with the output_var, properly indexed.
        """
        df: pd.DataFrame = xd[self.input_vars].to_dataframe()
        ind: pd.Index = df.index
        res = design_matrices(
            f"0 + {self.formula}", df, na_action="pass"
        ).common.as_dataframe()
        varname = res.columns[0]
        return res.rename(columns={varname: self.output_var}).set_index(ind).to_xarray()


@dataclass(eq=True, frozen=True)
class VariableDifference(VariableSingleEdge):
    """A Variable class for the 1-year growth transformation function. There is room for generalization here.

    Parameters
    ----------
    output_var : str
        Name of the output variable
    subset : int
        Endogenous simulation requires that all variables are fully specified in a circular fashion.
        At the same time, there cannot be any circular definitions in the transformation step, nor in the forecast step.
        If the variable is subset == 0, it is estimated/calculated in the transformation step, if it is 1, it is in the forecast step.
    input_var : str
        Name of the input variable
    type : Literal["growth]
        Currently only supports "growth" as case. Any temporal difference function could in principle be supported here.
    """

    type: Literal["growth"]

    def __post_init__(self):
        if self.type not in ["growth"]:
            raise NotImplementedError(
                f"{self.__repr__}: {self.type} is not implemented."
            )

    def calc(self, xd: xarray.Dataset) -> xarray.DataArray:
        """A growth function that takes a single input variable and outputs the growth from t-1 to t. This calculates the exact growth, and
        not growth based on log difference. Uses xarray internal functions.

        Warning! Divide by zero is a possibility here.

        Parameters
        ----------
        xd : xarray.Dataset
            A xarray.Dataset, often `EndogenousSystem._xa` (simulation data), or the `EndogenousSystem._past`.

        Returns
        -------
        xarray.DataArray
            Returns a DataArray with the output_var, properly indexed.
        """
        arr: xarray.DataArray = xd[self.input_var]
        match self.type:
            case "growth":
                res = (arr - arr.shift(ds=1)) / arr.shift(ds=1)
                res.name = self.output_var
                return res
            case _:
                raise NotImplementedError


@dataclass(eq=True, frozen=True)
class VariableRolling(VariableSingleEdge):
    """A Variable class for rolling time-series functions. Rolling means and sums, as well as exponentially weighted moving windows are supported.
    See `xarray.Dataset.rolling` and `xarray.Dataset.rolling_exp` for details.

    Parameters
    ----------
    output_var : str
        Name of the output variable
    subset : int
        Endogenous simulation requires that all variables are fully specified in a circular fashion.
        At the same time, there cannot be any circular definitions in the transformation step, nor in the forecast step.
        If the variable is subset == 0, it is estimated/calculated in the transformation step, if it is 1, it is in the forecast step.
    input_var : str
        Name of the input variable
    window : int
        The window size in time-units.
    fun : Literal["mean", "sum]
        The aggregation function (rolling mean or rolling sum).
    window_type : Literal["normal", "span", "com", "halflife", "alpha"]
        "normal" is equally weighted. See `pandas.DataFrame.ewm` for details on the rest.
    """

    window: int
    fun: Literal["mean", "sum"]
    window_type: Literal["normal", "span", "com", "halflife", "alpha"]

    def __post_init__(self):
        if self.fun not in ["mean", "sum"]:
            raise NotImplementedError(
                f"{self.__repr__}: {self.fun} is not implemented."
            )
        if self.window_type not in ["normal", "span", "com", "halflife", "alpha"]:
            raise NotImplementedError(
                f"{self.__repr__}: {self.window_type} is not implemented."
            )

    def calc(self, xd: xarray.Dataset) -> xarray.DataArray:
        """Calculates the temporally rolling transformations given an xarray.Dataset that includes the input_var. Uses xarray internal functions.

        Parameters
        ----------
        xd : xarray.Dataset
            A xarray.Dataset, often `EndogenousSystem._xa` (simulation data), or the `EndogenousSystem._past`.

        Returns
        -------
        xarray.DataArray
            Returns a DataArray with the output_var, properly indexed.

        Raises
        ------
        NotImplementedError
            `fun` must be "mean" or "sum". Any other aggregation function is not supported.
        NotImplementedError
            `window_type` must be "normal", "span", "com", "halflife", or "alpha". Other types are not supported.
        """
        match self.window_type:
            case "normal":
                roller: DataArrayRolling | RollingExp = xd[self.input_var].rolling(
                    ds=self.window
                )
            case "span" | "com" | "halflife" | "alpha":
                roller: DataArrayRolling | RollingExp = xd[self.input_var].rolling_exp(
                    ds=self.window, window_type=self.window_type
                )
            case _:
                raise NotImplementedError
        match self.fun:
            case "mean":
                res: xarray.DataArray = roller.mean()
            case "sum":
                res: xarray.DataArray = roller.sum()
            case _:
                raise NotImplementedError
        res.name: str = self.output_var
        return res


@dataclass(eq=True, frozen=True)
class VariableLag(VariableSingleEdge):
    """A Variable class for lagged variables (temporal offset). Uses `xarray.DataArray.shift`.

    Parameters
    ----------
    output_var : str
        Name of the output variable
    subset : int
        Endogenous simulation requires that all variables are fully specified in a circular fashion.
        At the same time, there cannot be any circular definitions in the transformation step, nor in the forecast step.
        If the variable is subset == 0, it is estimated/calculated in the transformation step, if it is 1, it is in the forecast step.
    input_var : str
        Name of the input variable
    num_lag : int
        How many time-units to offset. E.g., 1 would lag a time-series 1 time-unit compared to the input_var.
    """

    input_var: str
    num_lag: int

    def calc(self, xd: xarray.Dataset) -> xarray.DataArray:
        """Calculates the lagged output variable given a xarray.Dataset that includes the input variable.

        Parameters
        ----------
        xd : xarray.Dataset
            A xarray.Dataset, often `EndogenousSystem._xa` (simulation data), or the `EndogenousSystem._past`.

        Returns
        -------
        xarray.DataArray
            Returns a DataArray with the output_var, properly indexed.
        """
        res: xarray.DataArray = xd[self.input_var].shift(ds=self.num_lag)
        res.name = self.output_var
        return res
