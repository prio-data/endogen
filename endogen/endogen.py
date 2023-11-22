from .config import InputModel, Differences, Lags, Rolling, Transform
from .variables import (
    Variable,
    VariableLag,
    VariableRolling,
    VariableDifference,
    VariableTransform,
)
from .tools import measure, flatten, flatten_recursive
from .data_utilities import (
    read_input_data,
    drop_superfluous,
    drop_missing_units,
    generate_comparison_report,
)

from .utilities import PanelUnits
from .adapter_mlforecast import forecast_mlforecast

import xarray
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import collections

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from mlforecast.forecast import MLForecast
from mlforecast.utils import PredictionIntervals

import os

from formulae import design_matrices
from typing import Optional, Sequence, Iterable, Any, Mapping, Tuple, Literal

import logging

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSchedule:
    delta_t: int
    schedule: Iterable[str | Iterable[str]]


class ModelController:
    """A controller for organizing and scheduling models."""

    def __init__(self):
        self._models = []
        self._graph = nx.DiGraph()

    def add_models(self, models: InputModel | Sequence[InputModel]) -> None:
        """Adds a model to the system.

        Parameters
        ----------
        model : VariableModel
            Any model type supported by the VariableModel class.
        """
        if isinstance(models, InputModel):
            models = [models]

        output_vars = [m.output_var for m in self._models]
        new_models = [m for m in models if m.output_var not in output_vars]
        models_already_exists = [
            m.output_var for m in models if m.output_var in output_vars
        ]
        if len(models_already_exists) > 0:
            log.warning(
                f"Models for variables: {models_already_exists} are already loaded in the system. Please remove before adding."
            )

        self._models = [*self._models, *new_models]
        self._models_to_graph()

    @property
    def models(self) -> Sequence[InputModel]:
        return self._models

    def plot(self, path: str = None) -> None:
        fig = plt.figure()
        nx.draw(self._graph, with_labels=True, pos=nx.multipartite_layout(self._graph))
        if path != None:
            plt.savefig(path)
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)

    def _models_to_graph(self) -> None:
        def test_for_cyclic_graph(graph, edges) -> None:
            T = graph.copy()
            for edge in edges:
                T.add_edge(*edge)
                assert nx.is_directed_acyclic_graph(
                    T
                ), f"Adding edge: {edge} would introduce cyclic graphs."

        variable_types: Sequence[str] = ["lags", "differences", "rolling", "transforms"]

        variables = []
        for var_type in variable_types:
            for model in self.models:
                variable_recipies = getattr(model, var_type)
                if len(variable_recipies) > 0:
                    vars = [recip.get_variables() for recip in variable_recipies]
                    variables.append(vars)
        variables: Sequence[Variable] = flatten_recursive([variables, self.models])

        if set([var.subset for var in variables]) != {0, 1}:
            raise ValueError(
                f'Models must contain subsets 0 ("before forecast") and 1 ("after forecast"), and not any others.'
            )

        # There cannot be cycles before the forecast or during the forecast (each must be a DAG)
        for subset in [0, 1]:
            edges = flatten_recursive(
                [var.edges for var in variables if var.subset == subset]
            )
            test_for_cyclic_graph(self._graph, edges)

        self._graph.add_edges_from(flatten_recursive([var.edges for var in variables]))

        nodes = [var.node for var in variables]
        self._graph.add_nodes_from(nodes)

        self.prepare_nodes = [
            var.node[0] for var in variables if not isinstance(var, InputModel)
        ]
        self.derived_nodes = [var.node[0] for var in variables if var.subset == 0]
        self.forecast_nodes = [var.node[0] for var in variables if var.subset == 1]

        # There cannot be any variables without a model.
        empty_nodes: Sequence[str] = [
            n for n, v in self._graph.nodes(data=True) if v == {}
        ]
        if len(empty_nodes) > 0:
            raise ValueError(f"A model is missing for these variables: {empty_nodes}")

        # The complete graph should be completely cyclic (all nodes should have a cycle)
        # Perhaps put this into a try-except to catch the error more gracefully
        for node in list(self._graph.nodes):
            nx.find_cycle(self._graph, node)

    @property
    def _model_schedule(self) -> Tuple[ModelSchedule, ModelSchedule]:
        """Returns a schedule generator used to process variables in the correct order."""
        t0: ModelSchedule = ModelSchedule(
            delta_t=0,
            schedule=list(
                nx.topological_generations(self._graph.subgraph(self.derived_nodes)),
            ),
        )
        t1: ModelSchedule = ModelSchedule(
            delta_t=1,
            schedule=list(
                nx.topological_generations(self._graph.subgraph(self.forecast_nodes)),
            ),
        )
        return t0, t1


@dataclass
class EndogenousSystem:
    """An endogenous panel-data system of models/nodes with associated methods for correct scheduling of model forecasts.

    Parameters
    ----------
    input_data : str or pandas.DataFrame
        Panel data (or path to data) that includes all variables required by the forecasting system (and possibly fitting of models).
    time_var : str
        The variable name indicating time in input_data.
    unit_var : str
        The variable name indicating units in input_data.
    nsim : int
        The number of independent simulations of the endogenous system.
    start: int
        The number on the same scale as time_var when forecasting should start.
    end : int
        The number on the same scale as time_var when forecasting should end.
    vars : Optional[Sequence[str]]
        A subset of variables in input_data. Defaults to all variables in input_data.
    include_past_n : Optional[int]
        How much of the past to include when fitting statistical models.
    """

    input_data: str | os.PathLike | pd.DataFrame
    time_var: str
    unit_var: str
    nsim: int
    end: int
    vars: Optional[Sequence[str]] = field(default_factory=list)
    start: Optional[int] = None
    include_past_n: Optional[int] = None

    def __repr__(self):
        return f"EndogenousSystem({self._xa})"

    def __post_init__(self):
        self.pnames = PanelUnits(self.time_var, self.unit_var)

        # Read input-data
        self.input_data = read_input_data(self.input_data)

        # Use variables in the input-data unless specified
        if len(self.vars) == 0:
            self.vars = [
                var
                for var in self.input_data.columns
                if var not in [self.time_var, self.unit_var]
            ]
        else:
            self.vars = [
                var for var in self.vars if var not in [self.time_var, self.unit_var]
            ]

        if self.start == None:
            self.start = self.input_data[self.pnames.time_var].max() + 1
        
        self._last_train = self.start - 1

        data_to_xarray = self.input_data[
            self.input_data[self.pnames.time_var] < self.start
        ]

        report = generate_comparison_report(
            data_to_xarray,
            time_var=self.pnames.time_var,
            unit_var=self.pnames.unit_var,
            alternative_time_comparison=self._last_train,
        )

        self.missing_units = set().union(
            *report.loc[slice(self._last_train - self.include_past_n, self._last_train)][
                "missing"
            ].tolist()
        )
        if len(self.missing_units) > 0:
            log.warning(
                f"The following units were removed to attain a balanced dataset over {self.include_past_n} years: {self.missing_units}."
            )
        data_to_xarray = data_to_xarray[
            ~data_to_xarray[self.pnames.unit_var].isin(self.missing_units)
        ]

        data_to_xarray = drop_superfluous(
            data_to_xarray,
            time_var=self.pnames.time_var,
            unit_var=self.pnames.unit_var,
            alternative_time_comparison=self._last_train,
        )

        data_to_xarray = drop_missing_units(
            data_to_xarray,
            time_var=self.pnames.time_var,
            unit_var=self.pnames.unit_var,
            alternative_time_comparison=self._last_train,
        )

        data_to_xarray = data_to_xarray.rename(columns=self.pnames.to_dict()).set_index(
            self.pnames.internal_index
        )

        self._past = data_to_xarray[self.vars].dropna().to_xarray()
        self._past = self._past.sel(
            ds=slice(self._last_train - self.include_past_n, self._last_train)
        )
        del data_to_xarray
        # Initialize the model-controller
        self.models = ModelController()


    @classmethod
    def _make_container(
        cls,
        vars: Sequence[str],
        nsim: int,
        unit_index: pd.Index,
        time_index: pd.Index,
        ones: bool = False,
    ):
        nvar = len(vars)
        nunit = len(unit_index)
        ntime = len(time_index)

        if ones:
            arr = np.ones(shape=(nvar, ntime, nunit, nsim), dtype=np.float32)
        else:
            arr = np.zeros(shape=(nvar, ntime, nunit, nsim), dtype=np.float32)

        return xarray.DataArray(
            data=arr,
            dims=["vars", "ds", "unique_id", "sim"],
            coords={
                "vars": vars,
                "ds": time_index,
                "unique_id": unit_index,
            },
        ).to_dataset(dim="vars")

    def create_forecast_container(self):
        # To update self._past and self.vars with any transformations in self.models.models
        self.prepare_data()

        time_index = pd.Index(
            range(self._last_train - self.include_past_n, self.end), name="ds"
        )
        unit_index, _ = self._past.indexes.values()

        self._xa = self._make_container(
            vars=self.vars,
            nsim=self.nsim,
            unit_index=unit_index,
            time_index=time_index,
        )

        if isinstance(self.include_past_n, int) and self.include_past_n > 0:
            data_to_include = []
            single_slice = self._past.sel(ds=slice(time_index.start, time_index.stop))
            for _ in itertools.repeat(None, self.nsim):
                data_to_include.append(single_slice)
            data_to_include = xarray.concat(data_to_include, dim="sim").transpose(
                "ds", "unique_id", "sim"
            )
            self.update_sim(data_to_include)

    def prepare_data(self):
        t0, t1 = self.models._model_schedule

        for node_schedule in t0.schedule:
            for node in node_schedule:
                if isinstance(node, str) and node in self.models.prepare_nodes:
                    self._past = xarray.merge(
                        [
                            self._past,
                            self.models._graph.nodes[node]["model"].calc(xd=self._past),
                        ]
                    )

        # for node_schedule in t1.schedule:
        #     for node in node_schedule:
        #         if isinstance(node, str) and node in self.models.prepare_nodes:
        #             self._past = xarray.merge(
        #                 [
        #                     self._past,
        #                     self.models._graph.nodes[node]["model"].calc(xd=self._past),
        #                 ]
        #             )

        self._past = self._past.to_dataframe().dropna().to_xarray()
        self.vars = list(self._past.keys())

    def fit_models(self):
        for model in self.models.models:
            if isinstance(model.model, str):
                pass
            if isinstance(model.model, BaseEstimator):
                df = self._past.to_dataframe()
                y, X = df[model.output_var], df[model.input_vars]
                model.model.fit(X, y)
            if isinstance(model.model, MLForecast):
                data_variables = list(
                    itertools.chain([model.output_var], model.input_vars)
                )
                df = self._past.to_dataframe()[data_variables]
                df = df.rename(columns={model.output_var: "y"})
                df.reset_index(inplace=True)
                model.model.fit(
                    df,
                    static_features=[],
                    prediction_intervals=PredictionIntervals(n_windows=4, h=1),
                )

    def simulate(self):
        levels = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
        t0, t1 = self.models._model_schedule
        for t in range(self.start, self.end):
            ds_index = (self._xa.ds.values == t).nonzero()[0][0]
            for schedules in [t0, t1]:
                for node_schedule in schedules.schedule:
                    for node in node_schedule:
                        model = self.models._graph.nodes[node]["model"]

                        input_vars = [
                            m.input_vars
                            for m in self.models.models
                            if m.output_var == node
                        ]
                        if len(input_vars) == 1:
                            input_vars = input_vars[0]
                        else:
                            del input_vars

                        match model:
                            case VariableTransform():
                                self._xa[node][ds_index] = (
                                    model.calc(xd=self._xa.sel(ds=t))
                                )[node]
                            case VariableDifference() | VariableLag() | VariableRolling():
                                self._xa[node] = model.calc(xd=self._xa)
                            case BaseEstimator():
                                for s in range(self.nsim):
                                    self._xa[node][ds_index, :, s] = model.predict(
                                        self._xa[input_vars].to_dataframe().loc[t, :, s]
                                    )
                            case MLForecast():
                                for s in range(self.nsim):
                                    self._xa[node][
                                        ds_index, :, s
                                    ] = forecast_mlforecast(
                                        t,
                                        s,
                                        model,
                                        self._xa,
                                        self.pnames,
                                        node,
                                        input_vars,
                                        levels,
                                    )
                            case str():
                                for s in range(self.nsim):
                                    df = self._xa[input_vars].to_dataframe().loc[t,:,s]
                                    ind = df.index
                                    res = design_matrices(f'0 + {model}', df, na_action = "pass").common.as_dataframe()
                                    varname = res.columns[0]
                                    self._xa[node][
                                        ds_index, :, s
                                    ] = res.rename(columns={varname: node}).set_index(ind)[node].to_xarray()
                            case _:
                                raise NotImplementedError(
                                    f"Model of type {type(model)} is not implemented."
                                )

    def update_sim(self, value: xarray.Dataset | xarray.DataArray):
        if isinstance(value, xarray.Dataset):
            vars = list(value.keys())
            t_index = [
                i for i, k in enumerate(self._xa.ds.values) if k in value.ds.values
            ]
            for var in vars:
                self._xa[var][t_index] = value[var]
        elif isinstance(value, xarray.DataArray):
            t_index = [
                i for i, k in enumerate(self._xa.ds.values) if k in value.ds.values
            ]
            self._xa[value.name][t_index] = value
        else:
            raise ValueError(
                "Value must be either a xarray.Dataset or an xarray.DataArray object"
            )

    def plot(
        self,
        var: str,
        unit: Optional[Sequence[int]] = None,
        start: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """Plot method for historical and forecasted data.

        Parameters
        ----------
        var : str
            Name of the variable you want to plot
        unit : Optional[Sequence[int]]
            List of subset of units you want to plot in facets. Plot will otherwise show global statistics.
        start : Optional[int]
            Alternative start time. Plot will otherwise start as early as possible with the data given.
        args :
            Other arguments to pass to seaborn.relplot
        kwargs : key, value pairings
            Dictionary of keyword arguments to pass to seaborn.relplot

        Returns
        -------
        seaborn.FacetGrid
            An object managing one or more subplots that correspond to conditional data subsets with convenient methods for batch-setting of axes attributes.
        """
        if start == None:
            start = self._past.coords["ds"].min()

        if unit == None:
            units = self._xa.coords["unique_id"]
        else:
            units = unit

        forecast = self._xa.sel(unique_id=units, ds=slice(self.start + 1, self.end))[
            var
        ]
        past = self._past.sel(unique_id=units, ds=slice(start, self.start + 1))[var]
        forecast = forecast.to_dataframe()
        past = past.to_dataframe()

        forecast["type"] = "forecast"
        past["type"] = "historical"
        past["sim"] = 0
        past = past.reset_index().set_index(["ds", "unique_id", "sim"])
        forecast = forecast.reset_index().set_index(["ds", "unique_id", "sim"])
        if unit == None:
            plot_object = sns.relplot(
                pd.concat([past, forecast]),
                x="ds",
                y=var,
                hue="type",
                errorbar="sd",
                kind="line",
                *args,
                **kwargs,
            )
        else:
            plot_object = sns.relplot(
                pd.concat([past, forecast]),
                x="ds",
                y=var,
                col="unique_id",
                hue="type",
                errorbar="sd",
                kind="line",
                *args,
                **kwargs,
            )
        for ax in plot_object.axes.flat:
            ax.axvline(x=self.start + 0.5, color="black", linestyle="--")
        return plot_object
