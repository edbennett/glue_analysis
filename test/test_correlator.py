#!/usr/bin/env python3

from copy import deepcopy
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
    DataInconsistencyError,
    FrozenError,
    VEVData,
    concatenate,
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
    return pd.DataFrame(
        {"Correlation": np.arange(CORRELATOR_DATA_LENGTH, dtype=float)},
        index=pd.MultiIndex.from_product(
            [
                range(1, LENGTH_MC_TIME + 1),
                range(1, LENGTH_TIME + 1),
                range(1, LENGTH_INTERNAL + 1),
                range(1, LENGTH_INTERNAL + 1),
            ],
            names=["MC_Time", "Time", "Internal1", "Internal2"],
        ),
    )


@pytest.fixture()
def filename() -> str:
    return "filename"


@pytest.fixture()
def vev_data() -> CorrelatorData:
    return pd.DataFrame(
        {"Vac_exp": np.arange(VEV_DATA_LENGTH, dtype=float)},
        index=pd.MultiIndex.from_product(
            [
                range(1, LENGTH_MC_TIME + 1),
                range(1, LENGTH_INTERNAL + 1),
            ],
            names=["MC_Time", "Internal"],
        ),
    )


@pytest.fixture()
def frozen_corr_ensemble(
    filename: str, corr_data: CorrelatorData, vev_data: VEVData
) -> CorrelatorEnsemble:
    return create_corr_ensemble(filename, corr_data, vev_data, frozen=True)


@pytest.fixture()
def unfrozen_corr_ensemble(
    filename: str, corr_data: CorrelatorData, vev_data: VEVData
) -> CorrelatorEnsemble:
    return create_corr_ensemble(filename, corr_data, vev_data, frozen=False)


def create_corr_ensemble(
    filename: str, corr_data: CorrelatorData, vev_data: VEVData, *, frozen: bool
) -> CorrelatorEnsemble:
    corr_ensemble = CorrelatorEnsemble(filename)
    corr_ensemble.correlators = corr_data
    corr_ensemble.vevs = vev_data
    if frozen:
        corr_ensemble.freeze()
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
    ("prop", "value"),
    [
        ("num_timeslices", LENGTH_TIME),
        ("num_internal", LENGTH_INTERNAL),
        ("num_samples", LENGTH_MC_TIME),
    ],
    ids=["num_timeslices", "num_internal", "num_samples"],
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


@pytest.mark.skip(reason="We currently don't support non-int Internal indexing.")
def test_correlator_ensemble_reports_correct_num_internal_for_other_types(
    unfrozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    # This is a relict from the old data structure and needs updating when
    # un-skipping this test:
    unfrozen_corr_ensemble.correlators[
        ["Internal1", "Internal2"]
    ] = unfrozen_corr_ensemble.correlators[["Internal1", "Internal2"]].map(
        lambda x: (x**2, str(x))  # make a non-integer type
    )
    assert unfrozen_corr_ensemble.num_internal == LENGTH_INTERNAL


def test_correlator_ensemble_reports_correct_num_samples_with_gaps(
    unfrozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    unfrozen_corr_ensemble.correlators.index = (
        unfrozen_corr_ensemble.correlators.index.set_levels(
            unfrozen_corr_ensemble.correlators.index.levels[0].map(
                lambda x: x**2  # such that max(MC_Time) != num_samples
            ),
            level="MC_Time",
        )
    )
    assert unfrozen_corr_ensemble.num_samples == LENGTH_MC_TIME


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
        frozen_corr_ensemble.get_numpy().ravel()
        == frozen_corr_ensemble.correlators["Correlation"].to_numpy()
    ).all()


def test_correlator_ensemble_returns_sorted_numpy_data(
    unfrozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    expected = unfrozen_corr_ensemble.correlators["Correlation"].to_numpy()
    unfrozen_corr_ensemble.correlators = unfrozen_corr_ensemble.correlators.sample(
        frac=1
    )
    assert (unfrozen_corr_ensemble.freeze().get_numpy().ravel() == expected).all()


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
        frozen_corr_ensemble.get_numpy_vevs().ravel()
        == frozen_corr_ensemble.vevs["Vac_exp"].to_numpy()
    ).all()


def test_correlator_ensemble_returns_sorted_numpy_data_for_vevs(
    unfrozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    expected = unfrozen_corr_ensemble.vevs["Vac_exp"].to_numpy()
    unfrozen_corr_ensemble.vevs = unfrozen_corr_ensemble.vevs.sample(frac=1)
    assert (unfrozen_corr_ensemble.freeze().get_numpy_vevs().ravel() == expected).all()


def test_correlator_ensemble_raises_for_subtract_without_vevs_present(
    frozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    del frozen_corr_ensemble._vevs  # noqa: SLF001
    with pytest.raises(
        ValueError, match="Can't subtract vevs that have not been read."
    ):
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
                corr_np[:, i, j] - vevs_np[i] * vevs_np[j]
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
    ("data", "ensemble_name"),
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
    "column_name",
    CorrelatorData.get_metadata()[None]["columns"].keys(),
)
def test_correlator_ensemble_freezing_fails_with_missing_column(
    unfrozen_corr_ensemble: CorrelatorEnsemble,
    column_name: str,
) -> None:
    unfrozen_corr_ensemble.correlators = unfrozen_corr_ensemble.correlators.drop(
        column_name, axis="columns"
    )
    with pytest.raises(pa.errors.SchemaError):
        unfrozen_corr_ensemble.freeze()


@pytest.mark.parametrize("column_name", VEVData.get_metadata()[None]["columns"].keys())
def test_correlator_ensemble_freezing_fails_with_missing_column_in_vevs(
    unfrozen_corr_ensemble: CorrelatorEnsemble,
    column_name: str,
) -> None:
    unfrozen_corr_ensemble.correlators = unfrozen_corr_ensemble.vevs.drop(
        column_name, axis="columns"
    )
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


def test_correlator_ensemble_freezing_fails_with_wrong_datatypes(
    unfrozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    unfrozen_corr_ensemble.correlators["Correlation"] = "str is surely the wrong dtype"
    with pytest.raises(pa.errors.SchemaError):
        unfrozen_corr_ensemble.freeze()


@pytest.mark.parametrize(
    "column_name",
    CorrelatorData.index.get_metadata()[None]["columns"].keys(),
)
def test_correlator_ensemble_freezing_fails_with_wrong_index_datatypes(
    unfrozen_corr_ensemble: CorrelatorEnsemble, column_name: str
) -> None:
    idx = unfrozen_corr_ensemble.correlators.index
    unfrozen_corr_ensemble.correlators.index = idx.set_levels(
        idx.levels[idx.names.index(column_name)].map(str),
        level=column_name,
    )
    with pytest.raises(pa.errors.SchemaError):
        unfrozen_corr_ensemble.freeze()


def test_correlator_ensemble_freezing_fails_with_wrong_datatypes_in_vevs(
    unfrozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    unfrozen_corr_ensemble.vevs["Vac_exp"] = "str is surely the wrong dtype"
    with pytest.raises(pa.errors.SchemaError):
        unfrozen_corr_ensemble.freeze()


@pytest.mark.parametrize(
    "column_name",
    VEVData.index.get_metadata()[None]["columns"].keys(),
)
def test_correlator_ensemble_freezing_fails_with_wrong_index_datatypes_vevs(
    unfrozen_corr_ensemble: CorrelatorEnsemble, column_name: str
) -> None:
    idx = unfrozen_corr_ensemble.vevs.index
    unfrozen_corr_ensemble.vevs.index = idx.set_levels(
        idx.levels[idx.names.index(column_name)].map(str),
        level=column_name,
    )
    with pytest.raises(pa.errors.SchemaError):
        unfrozen_corr_ensemble.freeze()


def test_correlator_ensemble_freezing_fails_if_internals_differ_in_content(
    unfrozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    # different from Internal1
    unfrozen_corr_ensemble.correlators.loc[0, "Internal2"] = 2
    with pytest.raises(pa.errors.SchemaError):
        unfrozen_corr_ensemble.freeze()


def test_correlator_ensemble_fails_if_indexing_rows_are_not_unique(
    unfrozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    idx = unfrozen_corr_ensemble.correlators.index.to_frame()
    # Carefully chosen: Internal1 must be indentical to Internal2 otherwise
    # another check is triggered, too.
    idx.iloc[0] = idx.iloc[4]
    unfrozen_corr_ensemble.correlators.index = pd.MultiIndex.from_frame(idx)
    message = (
        "Non-unique index, "
        "should be pa.errors.SchemaError but fails due to some incompatibility."
    )
    with pytest.raises(ValueError, match=message):
        unfrozen_corr_ensemble.freeze()


def test_correlator_ensemble_fails_if_indexing_rows_are_not_unique_vevs(
    unfrozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    idx = unfrozen_corr_ensemble.vevs.index.to_frame()
    idx.iloc[0] = idx.iloc[1]
    unfrozen_corr_ensemble.vevs.index = pd.MultiIndex.from_frame(idx)
    message = (
        "Non-unique index, "
        "should be pa.errors.SchemaError but fails due to some incompatibility."
    )
    with pytest.raises(ValueError, match=message):
        unfrozen_corr_ensemble.freeze()


def test_correlator_ensemble_fails_if_vevs_and_correlators_with_different_length_index(
    unfrozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    unfrozen_corr_ensemble.vevs = (
        unfrozen_corr_ensemble.vevs.reset_index(drop=False)
        .assign(MC_Time=-1)
        .drop_duplicates(subset=["MC_Time", "Internal"], keep="first")
        .set_index(["MC_Time", "Internal"])
    )
    # now vevs is shorter (but still has a consistent index not triggering other checks)
    with pytest.raises(DataInconsistencyError):
        unfrozen_corr_ensemble.freeze()


def test_correlator_ensemble_fails_if_vevs_and_correlators_with_different_index(
    unfrozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    unfrozen_corr_ensemble.vevs = (
        unfrozen_corr_ensemble.vevs.reset_index(drop=False)
        .assign(MC_Time=-1)
        .drop_duplicates(subset=["MC_Time", "Internal"], keep="first")
        .set_index(["MC_Time", "Internal"])
    )  # now MC_Time is different from any corresponding values in correlators
    unfrozen_corr_ensemble.correlators = unfrozen_corr_ensemble.correlators.loc(axis=0)[
        1, ...
    ]  # now correlators has the same length as vevs
    with pytest.raises(DataInconsistencyError):
        unfrozen_corr_ensemble.freeze()


def test_correlator_ensemble_can_freeze_without_validation(
    unfrozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    unfrozen_corr_ensemble.correlators = pd.DataFrame(
        ["rubbish that would fail validation"]
    )
    unfrozen_corr_ensemble.freeze(perform_expensive_validation=False)
    # Let's check this despite the fact that reaching this line without an
    # exception raised likely means we've done it:
    assert unfrozen_corr_ensemble.frozen


def test_correlator_ensemble_freezing_without_validation_still_performs_typecheck(
    unfrozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    unfrozen_corr_ensemble.correlators = "rubbish that would fail validation"
    with pytest.raises(TypeError):
        unfrozen_corr_ensemble.freeze(perform_expensive_validation=False)


### concatenate


def test_concatenate_raises_on_empty_argument() -> None:
    message = "You must give at least one correlator ensemble."
    with pytest.raises(ValueError, match=message):
        concatenate([])


def test_concatenate_returns_element_if_single_element_is_given() -> None:
    element = CorrelatorEnsemble("unimportant-filename")
    assert concatenate([element]) is element


def test_concatenate_concatenates_data_from_two_ensembles(
    frozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    second_ensemble = deepcopy(frozen_corr_ensemble)
    assert (
        (
            concatenate([frozen_corr_ensemble, second_ensemble]).correlators
            == pd.concat(
                [frozen_corr_ensemble.correlators, second_ensemble.correlators]
            )
        )
        .all()
        .all()
    )


def test_concatenate_preserves_first_filename(
    frozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    second_ensemble = deepcopy(frozen_corr_ensemble)
    second_ensemble.filename = "unimportant-other-name"
    # just to make sure and be explicit:
    assert frozen_corr_ensemble.filename != second_ensemble.filename

    assert (
        concatenate([frozen_corr_ensemble, second_ensemble]).filename
        == frozen_corr_ensemble.filename
    )


def test_concatenate_preserves_first_ensemble_name(
    frozen_corr_ensemble: CorrelatorEnsemble,
) -> None:
    second_ensemble = deepcopy(frozen_corr_ensemble)
    second_ensemble.ensemble_name = "unimportant-other-name"
    # just to make sure and be explicit:
    assert frozen_corr_ensemble.ensemble_name != second_ensemble.ensemble_name

    assert (
        concatenate([frozen_corr_ensemble, second_ensemble]).ensemble_name
        == frozen_corr_ensemble.ensemble_name
    )
