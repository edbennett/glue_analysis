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
