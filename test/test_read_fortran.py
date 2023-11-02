#!/usr/bin/env python3

from io import StringIO
from typing import TextIO

import pytest

from glue_analysis.readers.read_fortran import _read_correlators_fortran


@pytest.fixture()
def filename() -> str:
    return "testname.txt"


@pytest.fixture()
def trivial_file() -> StringIO:
    return StringIO("column-name")


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
