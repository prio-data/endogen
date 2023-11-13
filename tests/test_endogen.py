import pytest
from endogen.endogen import ModelSchedule, ModelController, PanelUnits, EndogenousSystem
from endogen.config import InputModel, Lags, Rolling, Differences, Transform
import pandas as pd


@pytest.fixture
def input_model() -> InputModel:
    return InputModel(
        stage="writing",
        output_var="out",
        input_vars=[
            "v1_l1",
            "v2_l1",
            "v1_rm2_l1",
            "v2_rm2_l1",
            "v1_gr_l1",
            "v2_gr_l1",
            "v1v2_l1",
        ],
        model="Hello World",
        lags=[Lags(1, ["v1", "v2", "v1_rm2", "v2_rm2", "v1_gr", "v2_gr", "v1v2"])],
        rolling=[Rolling(2, ["mean"], ["v1", "v2"])],
        differences=[Differences("growth", ["v1", "v2"])],
        transforms=[Transform("v1v2", ["v1", "v2"], "v1:v2")],
        subset=1,
    )


@pytest.fixture
def my_df() -> pd.DataFrame:
    d = {
        "v1": [3, 4, 5, 6, 1, 2, 4, 6, 3, 2],
        "v2": [7, 4, 3, 2, 3, 4, 5, 7, 5, 3],
        "out": [20, 24, 25, 18, 17, 15, 14, 13, 15, 14],
    }
    idx = pd.MultiIndex.from_arrays(
        [[1, 2, 3, 4, 5, 1, 2, 3, 4, 5], [10, 10, 10, 10, 10, 20, 20, 20, 20, 20]],
        names=("ds", "unique_id"),
    )
    return pd.DataFrame(data=d, index=idx)


@pytest.fixture
def my_system(my_df: pd.DataFrame) -> EndogenousSystem:
    return EndogenousSystem(
        input_data=my_df,
        time_var="ds",
        unit_var="unique_id",
        nsim=5,
        end=7,
        include_past_n=2,
    )


def test_add_model_to_system(my_system: EndogenousSystem, input_model: pd.DataFrame):
    assert len(my_system.models.models) == 0
    my_system.models.add_model(input_model)
    assert len(my_system.models.models) == 1
