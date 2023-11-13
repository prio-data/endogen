# Useage

Simulation of a system of interdependent models comes at the cost of needing extensive setup and configuration. There is not just one model to specify, there are N models to specify. Moreover, they must match in the specific sense that the *transformation* step and the *forecast* step are each directed acyclic graphs (DAGs), whilst the combination of these two steps must result in a perfect circle with no hanging variables.

ENDOGEN solves this through setting up the data models through extensive configuration schemas, with the option to write and store configuration in YAML and load with [hydra](https://hydra.cc/). The model controler in ENDOGEN also automatically solves sequencing of interdependent calculations by putting everything into a [NetworkX](https://networkx.org/) modeling graph, and solving the two steps of the graph using `networkx.topological_generations`.

The flowcharts below show an example of how this might look like.

```{mermaid}
flowchart
    transformation-- t+1 -->forecast
    forecast-->transformation
```

```{mermaid}
flowchart
    subgraph Transformation
    var1 -- rolling_mean3 --> var1_rm3
    var1_rm3 -- lag1 --> var1_rm3_l1
    var1 -- lag1 --> var1_l1
    var2 -- growth --> var2_gr
    var2_gr -- lag1 --> var2_gr_l1
    var2 -- lag1 --> var2_l1
    end
```

```{mermaid}
flowchart
    subgraph Forecast
    var1_l1 -- predict --> var1
    var1_rm3_l1 -- predict --> var1
    var2_gr_l1 -- predict --> var1
    var1_l1 -- predict --> var2
    var2_l1 -- predict --> var2
    end
```

ENDOGEN is using a fixed naming scheme for variable transformations:

    {base}_{suffix}{number} 

Multiple suffixes can be added with additional underscores, as shown in the example flowcharts above. See {func}`endogen.config.Lags` and {func}`endogen.config.Rolling`.

The exception to this rule are {func}`endogen.config.Transform`, since these are built from arbitrary functions. Here, the user must specify an `output_var` as name.

## Using configuration files

```{code} bash
.
├── config.yaml
├── global_config
│   └── simple_run.yaml
└── variables
    ├── growth_v1.yaml
    ├── intensity_level_v1.yaml
    ├── population_v1.yaml
    └── v2x_libdem_v1.yaml
```

When using .yaml configuration in ENDOGEN, you will have to create a configuration folder that looks like the above, with "global_config" and "variables" sub-folders, and a "config.yaml" in the root. The configuration can be loaded using the following code:

```{code} python
from hydra.utils import instantiate
import hydra

with hydra.initialize(version_base = "1.3", config_path="conf"):
    config = hydra.compose(config_name = "config")
    config = instantiate(config, _convert_="all")

config.global_config
config.variables
```

Here, `config_path` is the path to the config folder, in this case named "conf". The `config_name` is the name of the main/root config file "config.yaml".

### config.yaml

```{code} yaml
defaults:
   - global_config: simple_run
   - variables@_variable_dict.growth: growth_v1
   - variables@_variable_dict.intensity_level: intensity_level_v1
   - variables@_variable_dict.v2x_libdem: v2x_libdem_v1
   - variables@_variable_dict.psecprop: psecprop_v1
   - variables@_variable_dict.population: population_v1
   - _self_
   
_target_: endogen.config.Config   
variables: ${oc.dict.values:_variable_dict}
```

We see that the `\_target\_` of the "config.yaml" file is {func}`endogen.config.Config`, and it must comply with the schema of this python class. `Config` takes a {func}`endogen.config.GlobalSimConfig` and a list of {func}`endogen.config.InputModel` as input.

The "config.yaml" in the root folder must look something like what you see above, with each InputModel linked using this rather offputting syntax (that hopefully will be improved in later versions of hydra, see [here](https://github.com/facebookresearch/hydra/issues/1939) and [here](https://stackoverflow.com/questions/71052174/using-multiple-configs-in-the-same-group-to-interpolate-values-in-a-yaml-file)). The important thing here is that the strings after the colons refer to stems of file names inside the "global_config" and "variables" folders, respectively. If you want to add a InputModel, you can add a new "- variables@_variable_dict.{var_name}: {file_stem}" entry.

### Global configuration
```{code} yaml
_target_: endogen.config.GlobalSimConfig
input_data: data/cy_data_static_test.csv
time_var: year
unit_var: gwcode
nsim: 10
end: 2030
include_past_n: 5
vars: [grwth, gdppc, population, psecprop, intensity_level, v2x_libdem]
```

"simple_run.yaml" is residing in the "global_config" subfolder, and has {func}`endogen.config.GlobalSimConfig` as`\_target\_`.

Here, the path (relative to the current working directory or absolute) to the input_data, the names of panel data dimensions, number of simulations, when to end simulations, how much past data to include when fitting models, and the variables to include from the input_data can be set up.

### Variables (InputModel)

```{code} yaml
_target_: endogen.config.InputModel
stage: writing
output_var: grwth
input_vars: [gdppc_l1, grwth_rm4_l1, intensity_level_hlm8_l1, psecprop_l1]
model: 
  _target_ : mlforecast.MLForecast
  models:
    - _target_: sklearn.linear_model.LinearRegression
    - _target_: sklearn.ensemble.RandomForestRegressor
  freq: 1
  num_threads: 4
lags:
  - _target_: endogen.config.Lags
    num_lag: 1
    input_vars: [gdppc, grwth_rm4, intensity_level_hlm8, psecprop]
rolling:
  - _target_: endogen.config.Rolling
    window: 4
    funs: [mean]
    input_vars: [grwth]
  - _target_: endogen.config.Rolling
    window: 8
    window_type: halflife
    funs: [mean]
    input_vars: [intensity_level]
transforms:
  - _target_: endogen.config.Transform
    output_var: gdppc
    input_vars: [gdppc_l1, grwth]
    formula: np.add(gdppc_l1, gdppc_l1*grwth)
    after_forecast: True 
```

The {func}`endogen.config.InputModel` is the way to define a statistical variable model in ENDOGEN.

The above example shows how the configuration .yaml file for one variable "grwth" might look like. The "\_target\_" refers to a Python class that will be instantiated by `hydra` when loading the configuration. We can see that this configuration file is targeting the {func}`endogen.config.InputModel` class, and must follow the schema of that class to be valid.

The `output_var` is the name of the variable, and should be found for past observations in the `input_data` that is set in {func}`endogen.config.GlobalSimConfig`.

The `input_vars` is a list of input variables to the statistical model used to fit and predict `output_var`. These must all be 1 time-unit lags. They must also either be fully defined as transformations taking the `output_var` as input, or taking other output variables defined in other {func}`endogen.config.InputModel` as input, or a combination of the two.

In the above example, "grwth" rely on two other InputModels: "intensity_level" and "psecprop". It also rely on "gddpc" being included in `input_data`. However, given an initial value, you can see that "gdppc" is calculated as a function of "gdppc_l1" and "grwth". This particular tranformation is done in the forecast step, after the forecast, instead of in the transformation step, as it needs the currently estimated "grwth" in its calculation.

In the transformation step, the lags and rolling variables, as well as any transforms where `after_forecast` is False, are calculated. The transformations must be possible to describe as directed acyclic graphs (DAGs). In the above example, "intensity_level" is used to calculate the rolling mean 8-year half-life, "intensity_level_hlm8". Note that the output variable name is implicitly given by the fixed naming scheme. "intensity_level_hlm8" then becomes input to calculating the 1-year lagged variable "intensity_level_hlm8_l1".

The model used in this example is a [mlforecast](https://github.com/Nixtla/mlforecast) model defined as an equally weighted ensemble of two [scikit-learn](https://scikit-learn.org/stable/index.html) models, a linear (OLS) model and a random forest, both with default specifications. As we are using an integer based time-variable, `freq` must be 1 (each step increase the year with 1). ENDOGEN will fit these models with conformal prediction, asking for ten different prediction intervals, building a 0.05-percentile stepwise histogram across the complete predictive distribution. When making forecasts, the model will then draw randomly from this distribution for each prediction (at each time-step, for each unit, for each independent simulation run).
