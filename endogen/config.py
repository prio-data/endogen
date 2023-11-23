# from pydantic.dataclasses import dataclass, Field
# from pydantic import validator
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Any, Mapping, Tuple

from .variables import (
    VariableTransform,
    VariableDifference,
    VariableRolling,
    VariableLag,
)


@dataclass(eq=True, frozen=True)
class Transform:
    """A schema for describing a variable transform and a factory for `variables.VariableTransform` variables.
    Can be used separately, or in conjunction with .yaml files and `hydra.initialize`, `hydra.compose`, and `hydra.utils.instantiate`.

    Parameters
    ----------
    output_var : str
        Name of the output variable
    input_vars : list[str]
        List of input variables needed to create the output variable
    formula : str
        A Wilkinson formula supported by `formulae`. See https://bambinos.github.io/formulae/notebooks/getting_started.html#User-guide.
    after_forecast : bool
        Endogenous simulation requires that all variables are fully specified in a circular fashion.
        At the same time, there cannot be any circular definitions in the transformation step, nor in the forecast step.
        If after_forecast is True, the variable is estimated/calculated in the forecast step.
    """

    output_var: str
    input_vars: List[str]
    formula: str
    after_forecast: bool = False

    def get_variables(self) -> VariableTransform:
        """Helper function to create a VariableTransform.

        Returns
        -------
        VariableTransform
        """
        if self.after_forecast:
            subset = 1
        else:
            subset = 0
        return VariableTransform(
            output_var=self.output_var,
            input_vars=self.input_vars,
            formula=self.formula,
            subset=subset,
        )


@dataclass(eq=True, frozen=True)
class Differences:
    """A schema for describing a growth variables and a factory for `variables.VariableDifference` variables.
    Can be used separately, or in conjunction with .yaml files and `hydra.initialize`, `hydra.compose`, and `hydra.utils.instantiate`.

    Will return output variables named the same as the input variables, only with "_gr" suffixed.

    Parameters
    ----------
    type : Literal["growth]
        Growth is currently the only difference function available. Note that dividing with zero is a possibility.
    input_vars : list[str]
        List of input variables to transform.
    """

    type: Literal["growth"]
    input_vars: List[str]

    def get_output_name(self, input_var: str) -> str:
        match self.type:
            case "growth":
                suffix = "_gr"
            case _:
                raise ValueError(f"Unknown Differences.type: {self.type}")
        return f"{input_var}{suffix}"

    def get_variables(self) -> List[VariableDifference]:
        [
            VariableDifference(
                output_var=self.get_output_name(input_var),
                input_var=input_var,
                type=self.type,
                subset=0,
            )
            for input_var in self.input_vars
        ]


@dataclass(eq=True, frozen=True)
class Rolling:
    """A schema for describing variables with "rolling" transformations and a factory for `variables.VariableRolling` variables.
    Can be used separately, or in conjunction with .yaml files and `hydra.initialize`, `hydra.compose`, and `hydra.utils.instantiate`.

    Will return output variables named the same as the input variables, only with a suffix according to this scheme:

    {input_var}_{window_type_suffix}{fun_suffix}{window} where:

    =========== =======
    window_type suffix
    =========== =======
    normal      \_r
    span        \_rsp
    com         \_rc
    halflife    \_hl
    alpha       \_ral
    =========== =======

    ==== ======
    fun  suffix
    ==== ======
    mean m
    sum  s
    ==== ======

    Parameters
    ----------
    window : int
        The window size in time-units.
    funs : list[Literal["mean", "sum]]
        List of aggregation functions (rolling mean or rolling sum).
    input_vars : list[str]
        List of input variables to transform.
    window_type : Literal["normal", "span", "com", "halflife", "alpha"]
        "normal" is equally weighted. See `pandas.DataFrame.ewm` for details on the rest.
    """

    window: int
    funs: List[Literal["mean", "sum"]]
    input_vars: List[str]
    window_type: Literal["normal", "span", "com", "halflife", "alpha"] = "normal"

    def get_output_name(self, input_var: str, fun: str) -> str:
        match self.window_type:
            case "normal":
                suffix = "_r"
            case "span":
                suffix = "_rsp"
            case "com":
                suffix = "_rc"
            case "halflife":
                suffix = "_hl"
            case "alpha":
                suffix = "_ral"
            case _:
                raise ValueError(f"Unknown Rolling.window_type: {self.window_type}")
        match fun:
            case "mean":
                suffix = suffix + "m"
            case "sum":
                suffix = suffix + "s"
            case _:
                raise ValueError(f"Unknown function: {fun}, in Rolling.funs")
        return f"{input_var}{suffix}{self.window}"

    def get_variables(self) -> List[VariableRolling]:
        out = []
        for input_var in self.input_vars:
            for fun in self.funs:
                obj = VariableRolling(
                    output_var=self.get_output_name(input_var, fun),
                    input_var=input_var,
                    window=self.window,
                    fun=fun,
                    window_type=self.window_type,
                    subset=0,
                )
                out.append(obj)
        return out


@dataclass(eq=True, frozen=True)
class Lags:
    """A schema for describing a lagged variables and a factory for `variables.VariableLag` variables.
    Can be used separately, or in conjunction with .yaml files and `hydra.initialize`, `hydra.compose`, and `hydra.utils.instantiate`.

    Will return output variables named the same as the input variables, only with "_l{num_lag}" suffixed.

    Parameters
    ----------
    num_lag : int
        How many time-units to offset. E.g., 1 would lag a time-series 1 time-unit compared to the input_var.
    input_vars : list[str]
        List of input variables to transform.
    """

    num_lag: int
    input_vars: List[str]

    def get_output_names(self, input_var: str) -> str:
        return f"{input_var}_l{self.num_lag}"

    def get_variables(self) -> List[VariableLag]:
        return [
            VariableLag(
                output_var=self.get_output_names(input_var),
                input_var=input_var,
                num_lag=self.num_lag,
                subset=0,
            )
            for input_var in self.input_vars
        ]


@dataclass(eq=True, frozen=True)
class InputModel:
    """Configuration schema for statistical model of any variable, to be used in endogenous simulation.
    Can be used separately, or in conjunction with .yaml files and `hydra.initialize`, `hydra.compose`, and `hydra.utils.instantiate`.

    Endogenous simulation requires not only knowledge of the statistical model, but also of any other variable input in the model.
    Some of these might be statistical models on their own (e.g., an `InputModel`), whilst other variables might be variable transforms
    of various types (see the `variables` module). These models must be fully specified here. Note the naming conventions for variable transforms
    in the various `config` schemas. E.g., for referencing a 1-year lagged variable as input_var, you can put the non-lagged variable "var1" in `InputModel.lags`,
    and "var1_l1" in `InputModel.input_vars`. The `endogen.ModelController` will make sure variables are calculated in the correct sequence.

    Parameters
    ----------
    stage : Literal["writing", "evaluating", "production"]
        Information on at which development-stage the InputModel can be said to live in. Can be useful in larger production settings.
    output_var : str
        The name of the output variable in question.
    input_vars : list[str]
        List of input variables the output variable needs in its model.
    model : Any
        Any supported statistical (or otherwise) model class that can produce numerical output (forecasts) based on input data.
        Currently, that means any sklearn.base.BaseEstimator subclass or mlforecast.forecast.MLForecast
    lags : list[Lags]
        List of `config.Lags` necessary to build the `input_vars`.
    rolling : list[Rolling]
        List of `config.Rolling` necessary to build the `input_vars`.
    differences : list[Differences]
        List of `config.Differences` necessary to build the `input_vars`.
    transforms : list[Transforms]
        List of `config.Transforms` necessary to build the `input_vars`.
    subset : int
        Endogenous simulation requires that all variables are fully specified in a circular fashion.
        At the same time, there cannot be any circular definitions in the transformation step, nor in the forecast step.
        If the variable is subset == 0, it is estimated/calculated in the transformation step, if it is 1, it is in the forecast step.
    """

    stage: Literal["writing", "evaluating", "production"]
    output_var: str
    input_vars: List[str]
    model: Any
    lags: Optional[List[Lags]] = field(default_factory=list)
    rolling: Optional[List[Rolling]] = field(default_factory=list)
    differences: Optional[List[Differences]] = field(default_factory=list)
    transforms: Optional[List[Transform]] = field(default_factory=list)
    subset: int = field(default=1)

    @property
    def node(self) -> Tuple[str, Mapping[str, Any]]:
        """A node representation that interface well with NetworkX graphs.

        Returns
        -------
        Tuple[str, Mapping[str, Any]]
            A tuple where the first element is the output variable name ("node"), and the second element is a dictionary of node data.
        """
        return (self.output_var, {"model": self.model, "subset": self.subset})

    @property
    def edges(self) -> List[Tuple[str, str]]:
        """The edges between input variables and the output variable that interface well with NetworkX graphs.

        Returns
        -------
        List[Tuple[str, str]]
            A list of graph edges describing the links between the input_vars and the output_var.
        """
        return [(input_var, self.output_var) for input_var in self.input_vars]

@dataclass(eq=True, frozen=True)
class ExogenModel:
    """Configuration schema for statistical model of any variable, to be used in endogenous simulation.
    Can be used separately, or in conjunction with .yaml files and `hydra.initialize`, `hydra.compose`, and `hydra.utils.instantiate`.

    An ExogenModel variable is forecast data coming from somewhere else. Currenly only supports deterministic exogenous variables. It has to be a complete
    set of data for all units in the simulation system, from the start date of simulation to the end date.

    output_var : str
        The name of the output variable in question.
    exogen_data : str
        String path to .csv or .parquet file only including time_var, unit_var and output_var.
    subset : int
        This should just always be 1. Might remove this as an option.

    """
    output_var: str
    exogen_data: str
    subset: int = field(default=1)

    @property
    def node(self) -> Tuple[str, Mapping[str, Any]]:
        """A node representation that interface well with NetworkX graphs.

        Returns
        -------
        Tuple[str, Mapping[str, Any]]
            A tuple where the first element is the output variable name ("node"), and the second element is a dictionary of node data.
        """
        return (self.output_var, {"subset": self.subset})

@dataclass
class GlobalSimConfig:
    """Configuration schema for global simulation options.
    Can be used separately, or in conjunction with .yaml files and `hydra.initialize`, `hydra.compose`, and `hydra.utils.instantiate`.

    Parameters
    ----------
    input_data : str
        Path to input data in either .csv or .parquet file format. Used both for training and as initial values in simulation.
    time_var : str
        Name of the variable in the input_data indicating the time dimension. The variable must be integer type.
    unit_var : str
        Name of the variable in the input_data indicating the unit/spatial dimension. The variable must be integer type.
    nsim : int
        Number of independent simulations to run.
    end : int
        The time-unit to end simulation. Since these are fully described endogenous simulations, they can go indefinitely.
    include_past_n : int
        How much of the past to include when fitting statistical models.
    start : int
        The time-unit to start simulation. Must be an integer value found in the time_var series in the input_data.
    vars : list[str]
        The subset of variables in the input_data to include.
    """

    input_data: str
    time_var: str
    unit_var: str
    nsim: int
    end: int
    include_past_n: int
    start: Optional[int] = None
    vars: Optional[List[str]] = field(default_factory=list)


@dataclass
class Config:
    """Global configuration schema for the endogenous simulation.
    Can be used separately, or in conjunction with .yaml files and `hydra.initialize`, `hydra.compose`, and `hydra.utils.instantiate`.

    Parameters
    ----------
    global_config : GlobalSimConfig
        Global simulation configuration options
    variables : list[InputModel]
        List of configuration schema for input models to include in endogenous simulation. Note restrictions on circularity, etc.

    """

    _variable_dict: dict
    global_config: GlobalSimConfig
    variables: List[InputModel|ExogenModel] = field(default_factory=list)
