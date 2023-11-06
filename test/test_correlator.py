import pandas as pd
import pytest

from glue_analysis.correlator import CorrelatorData, CorrelatorEnsemble, VEVData

LENGTH_BIN_INDEX = 1
LENGTH_TIME = 2
LENGTH_OP_INDEX = 3
CORRELATOR_DATA_LENGTH = LENGTH_TIME * LENGTH_BIN_INDEX * LENGTH_OP_INDEX**2
VEV_DATA_LENGTH = LENGTH_BIN_INDEX * LENGTH_OP_INDEX


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
        .reset_index(drop=True)
        .assign(Correlation=range(CORRELATOR_DATA_LENGTH))
    )


@pytest.fixture()
def filename() -> str:
    return "filename"


@pytest.fixture()
def vev_data() -> CorrelatorData:
    return (
        pd.MultiIndex.from_product(
            [
                range(1, LENGTH_BIN_INDEX + 1),
                range(1, LENGTH_OP_INDEX + 1),
            ],
            names=["Bin_index", "Op_index"],
        )
        .to_frame()
        .reset_index(drop=True)
        .assign(Vac_exp=range(VEV_DATA_LENGTH))
    )


@pytest.fixture()
def corr_ensemble(
    filename: str, corr_data: CorrelatorData, vev_data: VEVData
) -> CorrelatorEnsemble:
    corr_ensemble = CorrelatorEnsemble(filename)
    corr_ensemble.correlators = corr_data
    corr_ensemble.vevs = vev_data
    corr_ensemble._frozen = True
    return corr_ensemble


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


@pytest.mark.parametrize(
    "prop,value",
    [("NT", LENGTH_TIME), ("num_ops", LENGTH_OP_INDEX), ("num_bins", LENGTH_BIN_INDEX)],
    ids=["NT", "num_ops", "num_bins"],
)
def test_correlator_ensemble_reports_correct_properties(
    corr_ensemble: CorrelatorEnsemble, prop: str, value: int
) -> None:
    assert getattr(corr_ensemble, prop) == value
    # The following violates One-assert-per-test rule but significantly
    # outweighs that on DRY.
    # Scramble as a second test:
    corr_ensemble.correlators = corr_ensemble.correlators.sample(frac=1)
    assert getattr(corr_ensemble, prop) == value


# We don't test the consistency checks at this point. They are extensive and
# rely on all those implicit conventions about the data structure we're about to
# change. Come back and test them when they are meaningful again.


def test_correlator_ensemble_returns_correctly_shaped_numpy(
    corr_ensemble: CorrelatorEnsemble,
) -> None:
    assert corr_ensemble.get_numpy().shape == (
        LENGTH_BIN_INDEX,
        LENGTH_TIME,
        LENGTH_OP_INDEX,
        LENGTH_OP_INDEX,
    )


def test_correlator_ensemble_returns_correct_numpy_data(
    corr_ensemble: CorrelatorEnsemble,
) -> None:
    assert (
        corr_ensemble.get_numpy().reshape(-1)
        == corr_ensemble.correlators["Correlation"].values
    ).all()


def test_correlator_ensemble_returns_sorted_numpy_data(
    corr_ensemble: CorrelatorEnsemble,
) -> None:
    expected = corr_ensemble.correlators["Correlation"].values
    corr_ensemble.correlators = corr_ensemble.correlators.sample(frac=1)
    assert (corr_ensemble.get_numpy().reshape(-1) == expected).all()


def test_correlator_ensemble_returns_correctly_shaped_numpy_vevs(
    corr_ensemble: CorrelatorEnsemble,
) -> None:
    assert corr_ensemble.get_numpy_vevs().shape == (
        LENGTH_BIN_INDEX,
        LENGTH_OP_INDEX,
    )


def test_correlator_ensemble_returns_correct_numpy_data_for_vevs(
    corr_ensemble: CorrelatorEnsemble,
) -> None:
    assert (
        corr_ensemble.get_numpy_vevs().reshape(-1)
        == corr_ensemble.vevs["Vac_exp"].values
    ).all()


def test_correlator_ensemble_returns_sorted_numpy_data_for_vevs(
    corr_ensemble: CorrelatorEnsemble,
) -> None:
    expected = corr_ensemble.vevs["Vac_exp"].values
    corr_ensemble.vevs = corr_ensemble.vevs.sample(frac=1)
    assert (corr_ensemble.get_numpy_vevs().reshape(-1) == expected).all()
