import pandas as pd
import pytest

from glue_analysis.correlator import CorrelatorData, CorrelatorEnsemble, VEVData

LENGTH_BIN_INDEX = 1
LENGTH_TIME = 2
LENGTH_OP_INDEX = 3
CORRELATOR_DATA_LENGTH = LENGTH_TIME * LENGTH_BIN_INDEX * LENGTH_OP_INDEX**2


@pytest.fixture()
def corr_data() -> CorrelatorData:
    return (
        pd.MultiIndex.from_product(
            [
                range(1, LENGTH_BIN_INDEX + 1),
                range(1, LENGTH_TIME + 1),
                range(1, LENGTH_OP_INDEX + 1),
                range(1, LENGTH_OP_INDEX + 1),
            ],
            names=["Bin_index", "Time", "Op_index1", "Op_index2"],
        )
        .to_frame()
        .assign(glue_bins=range(CORRELATOR_DATA_LENGTH))
    )


@pytest.fixture()
def filename() -> str:
    return "filename"


@pytest.fixture()
def corr_ensemble(filename: str, corr_data: CorrelatorData) -> CorrelatorEnsemble:
    corr_ensemble = CorrelatorEnsemble(filename)
    corr_ensemble.correlators = corr_data
    return corr_ensemble


@pytest.fixture()
def vev_data() -> CorrelatorData:
    return pd.DataFrame()


def test_correlator_ensemble_stores_filename() -> None:
    assert CorrelatorEnsemble("filename").filename == "filename"


@pytest.mark.xfail(reason="To be implemented later")
def test_correlator_ensemble_allows_to_set_correlators_as_garbage() -> None:
    with pytest.raises(ValueError):
        CorrelatorEnsemble(
            "filename"
        ).correlators = "garbage that will be forbidden later"


def test_correlator_ensemble_allows_to_set_correlators_with_correct_data(
    corr_data: CorrelatorData,
) -> None:
    CorrelatorEnsemble("filename").correlators = corr_data
    # reaching this point means it didn't raise


@pytest.mark.xfail(reason="To be implemented later")
def test_correlator_ensemble_allows_to_set_vevs_as_garbage() -> None:
    with pytest.raises(ValueError):
        CorrelatorEnsemble("filename").vevs = "garbage that will be forbidden later"


def test_correlator_ensemble_allows_to_set_vevs_with_correct_data(
    vev_data: VEVData,
) -> None:
    CorrelatorEnsemble("filename").vevs = vev_data
    # reaching this point means it didn't raise


def test_correlator_ensemble_reports_correct_NT(
    corr_ensemble: CorrelatorEnsemble,
) -> None:
    assert corr_ensemble.NT == LENGTH_TIME
