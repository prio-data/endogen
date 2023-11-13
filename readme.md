# ENDOGEN
**Dynamic simulation of socio-economic and political systems**

This site serves the code for the ENDOGEN dynamic endogenous simulator. It allows you to estimate a set of statistical models and use the predictions from these models as simulated input in the other models. At the core of the system is a model scheduler using [NetworkX](https://networkx.org/), a data-model based on [xarray](https://docs.xarray.dev/en/stable/index.html), and a set of options for variable transformations. ENDOGEN currently supports models supported by [Nixtla MLForecast](https://nixtla.github.io/mlforecast/). To set up and configure models and simulations, ENDOGEN is leveraging [hydra](https://hydra.cc/), making it easy to bootstrap and build extensive models in YAML.

ENDOGEN is currently under development. Expect breaking changes for each version. Use the tagged versions instead of the main branch.

ENDOGEN is developed through POLIMPACT. POLIMPACT is a research project funded by an ERC Advanced Grant running from Fall 2022 until Fall 2027, see https://www.prio.org/projects/polimpact and https://erc.easme-web.eu/?p=101055133.

Please see our webpage for further documentation.

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

## Installation

Requirements: A recently updated Linux or OS X operating system (tested with Ubuntu 20.04), or Windows with [WSL](https://learn.microsoft.com/en-us/windows/wsl/install).

1. Install [mamba](https://github.com/conda-forge/miniforge#mambaforge).

2. Add [conda-lock](https://github.com/conda/conda-lock) to the base environment.

``` console
$ mamba install --channel=conda-forge --name=base conda-lock
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