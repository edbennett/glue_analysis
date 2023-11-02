#!/usr/bin/env python3
from io import BytesIO
from typing import BinaryIO

import numpy as np
import pytest

from glue_analysis.readers.read_binary import HEADER_NAMES, _read_correlators_binary


@pytest.fixture()
def filename() -> str:
    return "testname.txt"


@pytest.fixture()
def trivial_file() -> BytesIO:
    return BytesIO()


@pytest.fixture()
def header() -> dict[str, int]:
    return {name: i for i, name in enumerate(HEADER_NAMES)}


def create_corr_file(header: dict[str, int]) -> BytesIO:
    memory_file = BytesIO()
    memory_file.write(
        np.array([header[name] for name in HEADER_NAMES], dtype=np.float64).tobytes()
    )
    memory_file.seek(0)

    return memory_file


@pytest.fixture()
def corr_file(header: dict[str, int]) -> BytesIO:
    return create_corr_file(header)


### Trivial behavior


def test_read_correlators_binary_records_filename(
    trivial_file: BinaryIO, filename: str
) -> None:
    answer = _read_correlators_binary(trivial_file, filename)
    assert answer.filename == filename


def test_read_correlators_binary_does_not_create_vev_if_not_given(
    trivial_file: BinaryIO, filename: str
) -> None:
    answer = _read_correlators_binary(trivial_file, filename)
    assert "vevs" not in dir(answer)


def test_read_correlators_binary_freezes_the_ensemble(
    trivial_file: BinaryIO, filename: str
) -> None:
    answer = _read_correlators_binary(trivial_file, filename)
    assert answer._frozen


### Actually functional behavior


def test_read_correlators_binary_makes_metadata_from_header_constant(
    filename: str,
) -> None:
    header = {name: 1 for name in HEADER_NAMES}
    corr_file = create_corr_file(header)
    answer = _read_correlators_binary(corr_file, filename)
    assert answer.metadata == header
