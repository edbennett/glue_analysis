from io import BytesIO
from typing import BinaryIO

import pytest

from glue_analysis.readers.read_binary import _read_correlators_binary


@pytest.fixture()
def filename() -> str:
    return "testname.txt"


@pytest.fixture()
def trivial_file() -> BytesIO:
    return BytesIO()


### Trivial behavior


def test_read_correlators_binary_records_filename(
    trivial_file: BinaryIO, filename: str
) -> None:
    answer = _read_correlators_binary(trivial_file, filename)
    assert answer.filename == filename
