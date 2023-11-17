#!/usr/bin/env python3

import itertools
from io import StringIO
from typing import Any, TextIO

import numpy as np
import pytest

from glue_analysis.correlator import CorrelatorData, VEVData
from glue_analysis.readers.read_fortran import _read_correlators_fortran


@pytest.fixture()
def filename() -> str:
    return "testname.txt"


@pytest.fixture()
def trivial_file() -> StringIO:
    return StringIO("column-name")


@pytest.fixture()
def columns() -> list[str]:
    return ["Bin_index", "Time", "Op1_index", "Op2_index", "Correlation"]


@pytest.fixture()
def vev_columns() -> list[str]:
    return ["Bin_index", "Op_index", "Vac_exp"]


@pytest.fixture(params=[(24, 200), (36, 4000), (48, 2400)])
def vev_metadata(request) -> dict[str, Any]:  # noqa: ANN001
    # pytest makes it non-trivial to get the type of `request`, so annotation is omitted
    lattice_temporal_extent, num_configs = request.param
    return {"NT": lattice_temporal_extent, "num_configs": num_configs}


def create_data(columns: list[str]) -> np.array:
    indexing_data = np.array(
        list(itertools.product(*[range(5) for col in columns[:-1]]))
    )
    return np.concatenate(
        [indexing_data, np.arange(indexing_data.shape[0]).reshape(-1, 1)], axis=1
    )


@pytest.fixture()
def data(columns: list[str]) -> np.array:
    return create_data(columns)


@pytest.fixture()
def vev_data(vev_columns: list[str]) -> np.array:
    return create_data(vev_columns)


@pytest.fixture()
def full_file(columns: list[str], data: np.array) -> StringIO:
    return create_full_file(columns, data)


def create_full_file(columns: list[str], data: np.array) -> StringIO:
    memory_file = StringIO()
    np.savetxt(memory_file, data, fmt=(data.shape[1] - 1) * ["%d"] + ["%f"])
    memory_file.seek(0)
    # This is not exactly the format that is used in the example data I've got
    # but it seems to be close enough for the current implementation not to
    # complain. Might need a better approximation at some point.
    return StringIO(" ".join(columns) + "\n" + memory_file.read())


@pytest.fixture()
def vev_file(vev_columns: list[str]) -> StringIO:
    data = create_data(vev_columns)
    return create_full_file(vev_columns, data)


### Trivial behavior


def test_read_correlators_fortran_records_filename(
    full_file: TextIO, filename: str
) -> None:
    answer = _read_correlators_fortran(full_file, filename)
    assert answer.filename == filename


def test_read_correlators_fortran_creates_channel_column(
    full_file: TextIO, filename: str
) -> None:
    answer = _read_correlators_fortran(full_file, filename)
    assert "channel" in answer.correlators.columns


def test_read_correlators_fortran_does_not_create_vev_if_not_given(
    full_file: TextIO, filename: str
) -> None:
    answer = _read_correlators_fortran(full_file, filename)
    assert not hasattr(answer, "vevs")


def test_read_correlators_fortran_sets_not_given_metadata_to_empty_dict(
    full_file: TextIO, filename: str
) -> None:
    answer = _read_correlators_fortran(full_file, filename)
    assert answer.metadata == {}


def test_read_correlators_fortran_freezes_the_ensemble(
    full_file: TextIO, filename: str
) -> None:
    answer = _read_correlators_fortran(full_file, filename)
    assert answer.frozen


### Actually functional behavior


def test_read_correlators_fortran_passes_on_metadata(
    full_file: TextIO, filename: str
) -> None:
    metadata = {"some": "thing"}
    answer = _read_correlators_fortran(full_file, filename, metadata=metadata)
    assert answer.metadata == metadata


def test_read_correlators_fortran_correctly_names_columns(
    full_file: TextIO, filename: str
) -> None:
    answer = _read_correlators_fortran(full_file, filename)
    assert set(answer.correlators.columns) == {*CorrelatorData.columns, "channel"}


def test_read_correlators_fortran_correctly_names_vev_columns(
    full_file: TextIO,
    filename: str,
    vev_file: TextIO,
    vev_metadata: dict[str, Any],
) -> None:
    answer = _read_correlators_fortran(
        full_file, filename, vev_file=vev_file, metadata=vev_metadata
    )
    assert set(answer.vevs.columns) == {*VEVData.columns, "channel"}


def test_read_correlators_fortran_preserves_data(
    full_file: TextIO, filename: str, data: np.array
) -> None:
    answer = _read_correlators_fortran(full_file, filename)
    assert (answer.correlators.drop("channel", axis=1).to_numpy() == data).all()


def test_read_correlators_fortran_preserves_normalised_data_in_vev(
    full_file: TextIO,
    filename: str,
    vev_data: np.array,
    vev_file: TextIO,
    vev_metadata: dict[str, Any],
) -> None:
    lattice_temporal_extent = vev_metadata["NT"]
    num_configs = vev_metadata["num_configs"]
    num_bins = 5
    normalisation = (lattice_temporal_extent * num_configs / num_bins) ** 0.5

    answer = _read_correlators_fortran(
        full_file,
        filename,
        vev_file=vev_file,
        metadata=vev_metadata,
    )

    normalised_vev_data = vev_data.astype("float64")
    normalised_vev_data[:, -1] /= normalisation

    assert (
        answer.vevs.drop("channel", axis="columns").to_numpy() == normalised_vev_data
    ).all()


def test_read_correlators_fortran_rejects_bad_cfg_count(
    full_file: TextIO,
    filename: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Number of configurations 13 is not divisible by number of samples .*",
    ):
        _read_correlators_fortran(full_file, filename, metadata={"num_configs": 13})


@pytest.mark.parametrize("metadata", [{}, {"num_configs": 200}, {"NT": 24}])
def test_read_correlators_fortran_rejects_vevs_with_missing_metadata(
    full_file: TextIO,
    filename: str,
    vev_file: TextIO,
    metadata: dict[str, Any],
) -> None:
    with pytest.raises(
        ValueError, match=".*must be specified to normalise VEVs correctly."
    ):
        _read_correlators_fortran(
            full_file, filename, vev_file=vev_file, metadata=metadata
        )
