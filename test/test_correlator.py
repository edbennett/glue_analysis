import pandas as pd
import pytest

from glue_analysis.correlator import CorrelatorData, CorrelatorEnsemble, VEVData


@pytest.fixture()
def corr_data() -> CorrelatorData:
    return pd.DataFrame()


@pytest.fixture()
def vev_data() -> CorrelatorData:
    return pd.DataFrame()


def test_correlator_ensemble_stores_filename() -> None:
    assert CorrelatorEnsemble("filename").filename == "filename"


# for documenting purposes, should be later amended to "does not allow"
def test_correlator_ensemble_allows_to_set_correlators_as_garbage() -> None:
    CorrelatorEnsemble("filename").correlators = "garbage that will be forbidden later"
    # reaching this point means it didn't raise


def test_correlator_ensemble_allows_to_set_correlators_with_correct_data(
    corr_data: CorrelatorData,
) -> None:
    CorrelatorEnsemble("filename").correlators = corr_data
    # reaching this point means it didn't raise


# for documenting purposes, should be later amended to "does not allow"
def test_correlator_ensemble_allows_to_set_vevs_as_garbage() -> None:
    CorrelatorEnsemble("filename").vevs = "garbage that will be forbidden later"
    # reaching this point means it didn't raise


def test_correlator_ensemble_allows_to_set_vevs_with_correct_data(
    vev_data: VEVData,
) -> None:
    CorrelatorEnsemble("filename").vevs = vev_data
    # reaching this point means it didn't raise
