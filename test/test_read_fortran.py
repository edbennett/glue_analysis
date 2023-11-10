#!/usr/bin/env python3

from io import StringIO
from typing import TextIO

import numpy as np
import pytest

from glue_analysis.readers.read_fortran import _read_correlators_fortran


@pytest.fixture()
def filename() -> str:
    return "testname.txt"


@pytest.fixture()
def trivial_file() -> StringIO:
    return StringIO("column-name")


@pytest.fixture()
def columns() -> list[str]:
    return ["MC_Time", "Time", "Internal1", "Internal2", "Correlation"]


@pytest.fixture()
def vev_columns() -> list[str]:
    return ["MC_Time", "Internal", "Vac_exp"]


def create_data(columns: list[str]) -> np.array:
    np.random.seed(42)
    data = np.random.randint(0, 10, (10, len(columns)))
    if "Internal1" in columns:
        data[:, columns.index("Internal1")] = np.roll(
            data[:, columns.index("Internal2")], shift=1
        )
    return data


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
    assert answer._frozen


### Actually functional behavior


def test_read_correlators_fortran_passes_on_metadata(
    full_file: TextIO, filename: str
) -> None:
    metadata = {"some": "thing"}
    answer = _read_correlators_fortran(full_file, filename, metadata=metadata)
    assert answer.metadata == metadata


def test_read_correlators_fortran_preserves_column_names(
    full_file: TextIO, filename: str, columns: list[str]
) -> None:
    answer = _read_correlators_fortran(full_file, filename)
    assert set(answer.correlators.columns) == set(columns + ["channel"])


def test_read_correlators_fortran_preserves_data(
    full_file: TextIO, filename: str, data: np.array
) -> None:
    answer = _read_correlators_fortran(full_file, filename)
    assert (answer.correlators.drop("channel", axis=1).values == data).all()


@pytest.mark.xfail(
    reason="Have to rethink data generation for this test. "
    "Fails due to new consistency checks.",
    strict=True,
)
def test_read_correlators_fortran_preserves_data_in_vev(
    full_file: TextIO, filename: str, vev_data: np.array, vev_file: TextIO
) -> None:
    answer = _read_correlators_fortran(full_file, filename, vev_file=vev_file)
    assert (answer.vevs.drop("channel", axis=1).values == vev_data).all()
