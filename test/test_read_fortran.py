#!/usr/bin/env python3

from copy import copy
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
    return ["this", "is-so-very-much", "a-random-string"]


@pytest.fixture()
def data(columns: list[str]) -> np.array:
    return np.random.randint(0, 10, (10, len(columns)))


@pytest.fixture()
def full_file(columns: list[str], data: np.array) -> StringIO:
    memory_file = StringIO()
    np.savetxt(memory_file, data)
    memory_file.seek(0)
    # This is not exactly the format that is used in the example data I've got
    # but it seems to be close enough for the current implementation not to
    # complain. Might need a better approximation at some point.
    return StringIO(" ".join(columns) + "\n" + memory_file.read())


### Trivial behavior


def test_read_correlators_fortran_records_filename(
    trivial_file: TextIO, filename: str
) -> None:
    answer = _read_correlators_fortran(trivial_file, filename)
    assert answer.filename == filename


def test_read_correlators_fortran_creates_channel_column(
    trivial_file: TextIO, filename: str
) -> None:
    answer = _read_correlators_fortran(trivial_file, filename)
    assert "channel" in answer.correlators.columns


def test_read_correlators_fortran_does_not_create_vev_if_not_given(
    trivial_file: TextIO, filename: str
) -> None:
    answer = _read_correlators_fortran(trivial_file, filename)
    assert "vevs" not in dir(answer)


def test_read_correlators_fortran_sets_not_given_metadata_to_empty_dict(
    trivial_file: TextIO, filename: str
) -> None:
    answer = _read_correlators_fortran(trivial_file, filename)
    assert answer.metadata == {}


def test_read_correlators_fortran_freezes_the_ensemble(
    trivial_file: TextIO, filename: str
) -> None:
    answer = _read_correlators_fortran(trivial_file, filename)
    assert answer._frozen


### Actually functional behavior


def test_read_correlators_fortran_passes_on_metadata(
    trivial_file: TextIO, filename: str
) -> None:
    metadata = {"some": "thing"}
    answer = _read_correlators_fortran(trivial_file, filename, metadata=metadata)
    assert answer.metadata == metadata


def test_read_correlators_fortran_preserves_column_names(
    full_file: TextIO, filename: str, columns: list[str]
) -> None:
    answer = _read_correlators_fortran(full_file, filename)
    assert set(answer.correlators.columns) == set(columns + ["channel"])


def test_read_correlators_fortran_preserves_data(
    full_file: TextIO, filename: str, data: list[str]
) -> None:
    answer = _read_correlators_fortran(full_file, filename)
    assert (answer.correlators.drop("channel", axis=1).values == data).all()


def test_read_correlators_fortran_preserves_data_in_vev(
    full_file: TextIO, filename: str, data: list[str]
) -> None:
    vev_file = copy(full_file)
    answer = _read_correlators_fortran(full_file, filename, vev_file=vev_file)
    assert (answer.vevs.drop("channel", axis=1).values == data).all()
