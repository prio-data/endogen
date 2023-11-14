# ENDOGEN
  **<div style="text-align: right">Dynamic simulation of socio-economic and political systems</div>**

This site serves as documentation for the ENDOGEN dynamic endogenous simulator. It allows you to estimate a set of statistical models and use the predictions from these models as simulated input in the other models. At the core of the system is a model scheduler using [NetworkX](https://networkx.org/), a data-model based on [xarray](https://docs.xarray.dev/en/stable/index.html), and a set of options for variable transformations. ENDOGEN currently supports models supported by [Nixtla MLForecast](https://nixtla.github.io/mlforecast/). To set up and configure models and simulations, ENDOGEN is leveraging [hydra](https://hydra.cc/), making it easy to bootstrap and build extensive models in YAML.

:::{note}
ENDOGEN is currently under development. Expect breaking changes for each version. Use the tagged versions instead of the main branch.
:::

## Minimal example without configuration files

```python
from endogen.endogen import EndogenousSystem
from dataclasses import asdict
from endogen.config import GlobalSimConfig, InputModel, Lags
from mlforecast.forecast import MLForecast
from sklearn.linear_model import LinearRegression

gc = GlobalSimConfig(input_data = "data/cy_data_static_test.csv",
                     time_var = "year",
                     unit_var = "gwcode",
                     nsim = 10,
                     end = 2050,
                     include_past_n = 30,
                     start = 2015,
                     vars = ['gdppc', 'psecprop'])

gdppc_model = InputModel(stage = "writing",
           output_var= "gdppc",
           input_vars = ["gdppc_l1", "psecprop_l1"],
           model = MLForecast(models = LinearRegression()),
           lags = [Lags(num_lag = 1, input_vars = ["gdppc", "psecprop"])])

edu_model = InputModel(stage = "writing",
           output_var= "psecprop",
           input_vars = ["gdppc_l1", "psecprop_l1"],
           model = MLForecast(models = LinearRegression()),
           lags = [Lags(num_lag = 1, input_vars = ["gdppc", "psecprop"])])

s = EndogenousSystem(**asdict(gc))
s.models.add_models([gdppc_model, edu_model])
s.create_forecast_container()
s.fit_models()
s.simulate()
s.plot("gdppc", unit = 475)
s.plot("psecprop", unit = 475)
```

## Example using configuration files

```python
from endogen.endogen import EndogenousSystem
from dataclasses import asdict
import hydra
from hydra.utils import instantiate

# Load configuration from yaml files
with hydra.initialize(version_base = "1.3", config_path="conf"):
    config = hydra.compose(config_name = "config")
    config = instantiate(config, _convert_="all")

# Make some adjustments (nice for fast experimentation)
config.global_config.end = 2050
config.global_config.include_past_n = 30
config.global_config.nsim = 10
config.global_config.vars=['gdppc', 'grwth', 'population', 'intensity_level', 'v2x_libdem', 'psecprop', "rgdp"]

# Instantiate a system
s = EndogenousSystem(**asdict(config.global_config))
# Add models of variables
s.models.add_models(config.variables)
# Build the required training data and simulation container.
s.create_forecast_container()
# Fit statistical models
s.fit_models()
# Dynamically simulate all variables based on the model controller schedule
s.simulate()

# Plot results of simulations for a particular country (or any panel-data you would like to use)
s.plot("gdppc", unit = 475)
s.plot("intensity_level", unit = 475)
s.plot("v2x_libdem", unit = 475)
s.plot("grwth", unit = 475)
s.plot("psecprop", unit = 475)
s.plot("population", unit = 475)
s.plot("rgdp", unit = 475)
```

## Installation

Requirements: A recently updated Linux or OS X operating system (tested with Ubuntu 20.04), or Windows with [WSL](https://learn.microsoft.com/en-us/windows/wsl/install).

1. Install [mamba](https://github.com/conda-forge/miniforge#mambaforge).

2. Add [conda-lock](https://github.com/conda/conda-lock) to the base environment.

``` console
$ mamba install --channel=conda-forge --name=base conda-lock
$ mamba update conda-lock
```

3. Install [git](https://git-scm.com/downloads).

4. Download our package from github:

```console
$ git clone https://github.com/prio-data/endogen
$ cd endogen
```

5. Create the virtual environment based on lock-file created from environment.yml

``` console
$ conda-lock install -n endogen_env  --mamba
$ mamba activate endogen_env
```

If your system is not linux-64 or osx-arm64 (M1,M2,M3, etc.), you will have to build conda-lock.yml with another platform before running `conda-lock install`.

6. Run poetry to add additional python package requirements.

```console
(endogen_env) $ poetry install
```

7. Optionally install graphviz to visualize graphs.

````{tab} Linux
```console
$ sudo apt-get install graphviz
```
````

````{tab} OS X
```console
$ brew install graphviz
```
````

## Contents

```{eval-rst}
.. toctree::
  :maxdepth: 1
 
  useage
  api
```