#!/usr/bin/env python3

from typing import Any

import numpy as np
import pandas as pd
import pandera as pa
import pyerrors as pe
import pytest
from pandera.typing import DataFrame as DataFrameType

from glue_analysis.correlator import (
    CorrelatorData,
    CorrelatorEnsemble,
    FrozenError,
    VEVData,
    to_obs_array,
)

LENGTH_MC_TIME = 5  # needs at least 5 or pe.Corr complains
LENGTH_TIME = 2
LENGTH_INTERNAL = 3
CORRELATOR_DATA_LENGTH = LENGTH_TIME * LENGTH_MC_TIME * LENGTH_INTERNAL**2
VEV_DATA_LENGTH = LENGTH_MC_TIME * LENGTH_INTERNAL
MC_TIME_AXIS = 0


@pytest.fixture()
def corr_data() -> CorrelatorData:
    return (
        pd.MultiIndex.from_product(
            [
                range(1, LENGTH_MC_TIME + 1),
                range(1, LENGTH_TIME + 1),
                range(1, LENGTH_INTERNAL + 1),
                range(1, LENGTH_INTERNAL + 1),
            ],
            names=["MC_Time", "Time", "Internal1", "Internal2"],
        )
        .to_frame()
        .reset_index(drop=True)
        .assign(Correlation=np.arange(CORRELATOR_DATA_LENGTH, dtype=float))
    )


@pytest.fixture()
def filename() -> str:
    return "filename"


@pytest.fixture()
def vev_data() -> CorrelatorData:
    return (
        pd.MultiIndex.from_product(
            [
                range(1, LENGTH_MC_TIME + 1),
                range(1, LENGTH_INTERNAL + 1),
            ],
            names=["MC_Time", "Internal"],
        )
        .to_frame()
        .reset_index(drop=True)
        .assign(Vac_exp=np.arange(VEV_DATA_LENGTH, dtype=float))
    )


@pytest.fixture()
def frozen_corr_ensemble(
    filename: str, corr_data: CorrelatorData, vev_data: VEVData
) -> CorrelatorEnsemble:
    return create_corr_ensemble(filename, corr_data, vev_data, True)


@pytest.fixture()
def unfrozen_corr_ensemble(
    filename: str, corr_data: CorrelatorData, vev_data: VEVData
) -> CorrelatorEnsemble:
    return create_corr_ensemble(filename, corr_data, vev_data, False)


def create_corr_ensemble(
    filename: str, corr_data: CorrelatorData, vev_data: VEVData, frozen: bool
) -> CorrelatorEnsemble:
    corr_ensemble = CorrelatorEnsemble(filename)
    corr_ensemble.correlators = corr_data
    corr_ensemble.vevs = vev_data
    corr_ensemble._frozen = frozen
    return corr_ensemble


def test_correlator_ensemble_stores_filename() -> None:
    assert CorrelatorEnsemble("filename").filename == "filename"


def test_correlator_ensemble_allows_to_set_correlators_as_garbage() -> None:
    # we decided to allow this because validation will be implemented in the
    # freeze method
    CorrelatorEnsemble("filename").correlators = "garbage that will be forbidden later"
    # reaching this point means it didn't raise


def test_correlator_ensemble_allows_to_set_correlators_with_correct_data(
    corr_data: CorrelatorData,
) -> None:
    CorrelatorEnsemble("filename").correlators = corr_data
    # reaching this point means it didn't raise


def test_correlator_ensemble_allows_to_set_vevs_as_garbage() -> None:
    # we decided to allow this because validation will be implemented in the
    # freeze method
    CorrelatorEnsemble("filename").vevs = "garbage that will be forbidden later"
    # reaching this point means it didn't raise


def test_correlator_ensemble_allows_to_set_vevs_with_correct_data(
    vev_data: VEVData,
) -> None:
    CorrelatorEnsemble("filename").vevs = vev_data
    # reaching this point means it didn't raise


@pytest.mark.parametrize(
    "prop,value",
    [
        ("NT", LENGTH_TIME),
        ("num_internal", LENGTH_INTERNAL),
        ("num_samples", LENGTH_MC_TIME),
    ],
    ids=["NT", "num_internal", "num_samples"],
)
def test_correlator_ensemble_reports_correct_properties(
    unfrozen_corr_ensemble: CorrelatorEnsemble, prop: str, value: int
) -> None:
    assert getattr(unfrozen_corr_ensemble, prop) == value
    # The following violates One-assert-per-test rule but significantly
    # outweighs that on DRY.
    # Scramble as a second test:
    unfrozen_corr_ensemble.correlators = unfrozen_corr_ensemble.correlators.sample(
        frac=1
    )
    assert getattr(unfrozen_corr_ensemble, prop) == value


# We don't test the consistency checks at this point. They are extensive and
# rely on all those implicit conventions about the data structure we're about to
# change. Come back and test them when they are meaningful again.


def test_correlator_ensemble_returns_correctly_shaped_numpy(
    frozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    assert frozen_corr_ensemble.get_numpy().shape == (
        LENGTH_MC_TIME,
        LENGTH_TIME,
        LENGTH_INTERNAL,
        LENGTH_INTERNAL,
    )


def test_correlator_ensemble_returns_correct_numpy_data(
    frozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    assert (
        frozen_corr_ensemble.get_numpy().reshape(-1)
        == frozen_corr_ensemble.correlators["Correlation"].values
    ).all()


def test_correlator_ensemble_returns_sorted_numpy_data(
    unfrozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    expected = unfrozen_corr_ensemble.correlators["Correlation"].values
    unfrozen_corr_ensemble.correlators = unfrozen_corr_ensemble.correlators.sample(
        frac=1
    )
    assert (unfrozen_corr_ensemble.freeze().get_numpy().reshape(-1) == expected).all()


def test_correlator_ensemble_returns_correctly_shaped_numpy_vevs(
    frozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    assert frozen_corr_ensemble.get_numpy_vevs().shape == (
        LENGTH_MC_TIME,
        LENGTH_INTERNAL,
    )


def test_correlator_ensemble_returns_correct_numpy_data_for_vevs(
    frozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    assert (
        frozen_corr_ensemble.get_numpy_vevs().reshape(-1)
        == frozen_corr_ensemble.vevs["Vac_exp"].values
    ).all()


def test_correlator_ensemble_returns_sorted_numpy_data_for_vevs(
    unfrozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    expected = unfrozen_corr_ensemble.vevs["Vac_exp"].values
    unfrozen_corr_ensemble.vevs = unfrozen_corr_ensemble.vevs.sample(frac=1)
    assert (
        unfrozen_corr_ensemble.freeze().get_numpy_vevs().reshape(-1) == expected
    ).all()


def test_correlator_ensemble_raises_for_subtract_without_vevs_present(
    frozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    del frozen_corr_ensemble._vevs
    with pytest.raises(ValueError):
        frozen_corr_ensemble.get_pyerrors(subtract=True)


def test_correlator_ensemble_returned_correlator_has_correct_averages(
    frozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    corr = frozen_corr_ensemble.get_pyerrors()
    corr_np = frozen_corr_ensemble.get_numpy().mean(axis=MC_TIME_AXIS)
    for i in range(LENGTH_INTERNAL):
        for j in range(LENGTH_INTERNAL):
            # not a perfect test: check for each entry of correlation matrix
            # that MC average equals the naive numpy result
            assert (corr_np[:, i, j] == corr.item(i, j).plottable()[1]).all()


def test_correlator_ensemble_returned_correlator_has_correct_subtracted_averages(
    frozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    corr = frozen_corr_ensemble.get_pyerrors(subtract=True)
    corr_np = frozen_corr_ensemble.get_numpy().mean(axis=MC_TIME_AXIS)
    vevs_np = frozen_corr_ensemble.get_numpy_vevs().mean(axis=MC_TIME_AXIS)
    for i in range(LENGTH_INTERNAL):
        for j in range(LENGTH_INTERNAL):
            # not a perfect test: check for each entry of correlation matrix
            # that MC average equals the naive numpy result
            assert (
                corr_np[:, i, j] - vevs_np[i] * vevs_np[j] / frozen_corr_ensemble.NT**2
                == corr.item(i, j).plottable()[1]
            ).all()


def test_correlator_ensemble_has_configurable_ensemble_name(
    corr_data: CorrelatorData,
) -> None:
    ensemble_name = "some-other-name"
    corr_ensemble = CorrelatorEnsemble(filename, ensemble_name=ensemble_name)
    corr_ensemble.correlators = corr_data
    assert corr_ensemble.freeze().get_pyerrors().item(0, 0).content[0][0].e_names == [
        ensemble_name
    ]


def test_correlator_ensemble_defaults_to_glue_bins_as_ensemble_name(
    frozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    assert frozen_corr_ensemble.get_pyerrors().item(0, 0).content[0][0].e_names == [
        "glue_bins"
    ]


### to_obs_array


@pytest.mark.parametrize(
    "data,ensemble_name",
    [
        (np.ones(10), "some-name"),
        (np.ones(10), "other-name"),
        (np.arange(10), "some-name"),
    ],
    ids=["trivial", "configurable-name", "other-data"],
)
def test_to_obs_array_works_on_one_dimensional_arrays(
    data: np.array, ensemble_name: str
) -> None:
    assert to_obs_array(data, ensemble_name) == pe.Obs([data], [ensemble_name])


def test_to_obs_array_works_on_two_dimensional_arrays() -> None:
    data = np.arange(20).reshape(10, 2)
    ensemble_name = "some-name"
    assert (
        to_obs_array(data, ensemble_name)
        == [
            pe.Obs([data[:, 0]], [ensemble_name]),
            pe.Obs([data[:, 1]], [ensemble_name]),
        ]
    ).all()


def test_to_obs_array_works_on_four_dimensional_arrays() -> None:
    data = np.arange(20).reshape(10, 2, 1, 1)
    ensemble_name = "some-name"
    assert (
        to_obs_array(data, ensemble_name)
        == np.asarray(
            [
                [[pe.Obs([data[:, 0, 0, 0]], [ensemble_name])]],
                [[pe.Obs([data[:, 1, 0, 0]], [ensemble_name])]],
            ]
        )
    ).all()


### freezing and validation


def test_correlator_ensemble_is_frozen_after_freezing(
    unfrozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    assert unfrozen_corr_ensemble.freeze().frozen


@pytest.mark.parametrize(
    "bad_data",
    ["garbage that is no reasonable data", 42, np.arange(10)],
    ids=["str", "int", "np.array"],
)
def test_correlator_ensemble_does_not_allow_garbage_correlators_on_freezing(
    unfrozen_corr_ensemble: CorrelatorEnsemble,
    bad_data: Any,  # noqa: ANN401
) -> None:
    unfrozen_corr_ensemble.correlators = bad_data
    with pytest.raises(TypeError):
        unfrozen_corr_ensemble.freeze()


@pytest.mark.parametrize(
    "bad_data",
    ["garbage that is no reasonable data", 42, np.arange(10)],
    ids=["str", "int", "np.array"],
)
def test_correlator_ensemble_does_not_allow_garbage_vevs_on_freezing(
    unfrozen_corr_ensemble: CorrelatorEnsemble,
    bad_data: Any,  # noqa: ANN401
) -> None:
    unfrozen_corr_ensemble.vevs = bad_data
    with pytest.raises(TypeError):
        unfrozen_corr_ensemble.freeze()


@pytest.mark.parametrize(
    "column_name", CorrelatorData.get_metadata()[None]["columns"].keys()
)
def test_correlator_ensemble_freezing_fails_with_missing_column(
    unfrozen_corr_ensemble: CorrelatorEnsemble, column_name: str
) -> None:
    unfrozen_corr_ensemble.correlators.drop(column_name, axis="columns", inplace=True)
    with pytest.raises(pa.errors.SchemaError):
        unfrozen_corr_ensemble.freeze()


@pytest.mark.parametrize("column_name", VEVData.get_metadata()[None]["columns"].keys())
def test_correlator_ensemble_freezing_fails_with_missing_column_in_vevs(
    unfrozen_corr_ensemble: CorrelatorEnsemble, column_name: str
) -> None:
    unfrozen_corr_ensemble.vevs.drop(column_name, axis="columns", inplace=True)
    with pytest.raises(pa.errors.SchemaError):
        unfrozen_corr_ensemble.freeze()


def test_correlator_ensemble_does_not_allow_alteration_after_freezing(
    frozen_corr_ensemble: CorrelatorEnsemble,
    corr_data: DataFrameType[CorrelatorData],
) -> None:
    with pytest.raises(FrozenError):
        frozen_corr_ensemble.correlators = corr_data


def test_correlator_ensemble_does_not_allow_alteration_of_vevs_after_freezing(
    frozen_corr_ensemble: CorrelatorEnsemble,
    vev_data: DataFrameType[VEVData],
) -> None:
    with pytest.raises(FrozenError):
        frozen_corr_ensemble.vevs = vev_data


@pytest.mark.parametrize(
    "column_name", CorrelatorData.get_metadata()[None]["columns"].keys()
)
def test_correlator_ensemble_freezing_fails_with_wrong_datatypes(
    unfrozen_corr_ensemble: CorrelatorEnsemble, column_name: str
) -> None:
    unfrozen_corr_ensemble.correlators = unfrozen_corr_ensemble.correlators.assign(
        **{column_name: "str is surely the wrong dtype"}
    )
    if column_name.startswith("Internal"):
        # anything is allowed for internal index
        unfrozen_corr_ensemble.freeze()
    else:
        with pytest.raises(pa.errors.SchemaError):
            unfrozen_corr_ensemble.freeze()


@pytest.mark.parametrize("column_name", VEVData.get_metadata()[None]["columns"].keys())
def test_correlator_ensemble_freezing_fails_with_wrong_datatypes_in_vevs(
    unfrozen_corr_ensemble: CorrelatorEnsemble, column_name: str
) -> None:
    unfrozen_corr_ensemble.vevs = unfrozen_corr_ensemble.vevs.assign(
        **{column_name: "str is surely the wrong dtype"}
    )
    if column_name.startswith("Internal"):
        # anything is allowed for Internal index
        unfrozen_corr_ensemble.freeze()
    else:
        with pytest.raises(pa.errors.SchemaError):
            unfrozen_corr_ensemble.freeze()
