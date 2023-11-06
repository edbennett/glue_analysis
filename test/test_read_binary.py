#!/usr/bin/env python3
from io import BytesIO
from typing import BinaryIO

import numpy as np
import pandas as pd
import pytest

from glue_analysis.readers.read_binary import (
    CORRELATOR_COLUMNS,
    CORRELATOR_INDEXING_COLUMNS,
    HEADER_NAMES,
    VEV_COLUMNS,
    VEV_INDEXING_COLUMNS,
    ParsingError,
    _read_correlators_binary,
)


@pytest.fixture()
def filename() -> str:
    return "testname.txt"


@pytest.fixture()
def header() -> dict[str, int]:
    return {name: i + 1 for i, name in enumerate(HEADER_NAMES)}


def columns_from_header(header: dict[str, int], vev: bool = False) -> pd.DataFrame:
    index_ranges = [
        range(1, header["Nbin"] + 1),  # Bin_index
        range(1, header["Nbl"] + 1),  # Blocking_index1
        range(1, header["Nop"] + 1),  # Op_index1
    ]
    if not vev:
        index_ranges += [
            range(1, header["Nbl"] + 1),  # Blocking_index2
            range(1, header["Nop"] + 1),  # Op_index2
            range(1, int(header["LT"] / 2 + 1) + 1),  # Time
        ]
    return (
        pd.MultiIndex.from_product(
            index_ranges,
            names=VEV_INDEXING_COLUMNS if vev else CORRELATOR_INDEXING_COLUMNS,
        )
        .to_frame()
        .reset_index(drop=True)
    )


def create_data(header: dict[str, int], vev: bool = False) -> np.array:
    return np.random.random(columns_from_header(header, vev=vev).shape[0])


@pytest.fixture()
def data(header: dict[str, int]) -> np.array:
    return create_data(header)


@pytest.fixture()
def vev_data(header: dict[str, int]) -> np.array:
    return create_data(header, vev=True)


def create_file(header: dict[str, int], data: np.array) -> BytesIO:
    memory_file = BytesIO()
    memory_file.write(
        np.array([header[name] for name in HEADER_NAMES], dtype=np.float64).tobytes()
    )
    memory_file.write(data.tobytes())
    memory_file.seek(0)
    return memory_file


@pytest.fixture()
def corr_file(header: dict[str, int], data: np.array) -> BytesIO:
    return create_file(header, data)


@pytest.fixture()
def vev_file(header: dict[str, int], vev_data: np.array) -> BytesIO:
    return create_file(header, vev_data)


@pytest.fixture()
def trivial_vevs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Bin_index": np.arange(10, dtype=np.float64),
            "Operator_index": np.ones(10, dtype=np.float64),
            "Blocking_index": np.ones(10, dtype=np.float64),
            "Vac_exp": np.ones(10, dtype=np.float64),
        }
    )


### Trivial behavior


def test_read_correlators_binary_records_filename(
    corr_file: BinaryIO, filename: str
) -> None:
    answer = _read_correlators_binary(corr_file, filename)
    assert answer.filename == filename


def test_read_correlators_binary_does_not_create_vev_if_not_given(
    corr_file: BinaryIO, filename: str
) -> None:
    answer = _read_correlators_binary(corr_file, filename)
    assert "vevs" not in dir(answer)


def test_read_correlators_binary_freezes_the_ensemble(
    corr_file: BinaryIO, filename: str
) -> None:
    answer = _read_correlators_binary(corr_file, filename)
    assert answer._frozen


### Actually functional behavior

#### Metadata


def test_read_correlators_binary_makes_metadata_from_header_constant(
    filename: str,
) -> None:
    header = {name: 1 for name in HEADER_NAMES}
    corr_file = create_file(header, create_data(header))
    answer = _read_correlators_binary(corr_file, filename)
    assert answer.metadata == header


def test_read_correlators_binary_makes_metadata_from_header_rising(
    filename: str,
) -> None:
    header = {name: i for i, name in enumerate(HEADER_NAMES)}
    corr_file = create_file(header, create_data(header))
    answer = _read_correlators_binary(corr_file, filename)
    assert answer.metadata == header


def test_read_correlators_binary_merges_header_with_metadata(
    corr_file: BinaryIO, filename: str, header: dict[str, int]
) -> None:
    metadata = {"some": "metadata"}
    answer = _read_correlators_binary(corr_file, filename, metadata=metadata)
    assert answer.metadata == header | metadata


def test_read_correlators_binary_raises_on_conflicting_metadata(
    corr_file: BinaryIO, filename: str
) -> None:
    metadata = {HEADER_NAMES[0]: "conflict with header info"}
    with pytest.raises(ParsingError):
        _read_correlators_binary(corr_file, filename, metadata=metadata)


def test_read_correlators_binary_raises_on_any_doubly_specified_metadata(
    corr_file: BinaryIO, filename: str, header: dict[str, int]
) -> None:
    metadata = {
        HEADER_NAMES[0]: header[HEADER_NAMES[0]]  # same as header but still forbidden
    }
    with pytest.raises(ParsingError):
        _read_correlators_binary(corr_file, filename, metadata=metadata)


#### VEVs


def test_read_correlators_binary_has_correct_columns_in_vev(
    corr_file: BinaryIO, filename: str, vev_file: BinaryIO
) -> None:
    answer = _read_correlators_binary(corr_file, filename, vev_file=vev_file)
    print(answer.vevs.columns)
    assert (answer.vevs.columns == VEV_COLUMNS).all()


def test_read_correlators_binary_has_indexing_columns_consistent_with_header_in_vev(
    corr_file: BinaryIO, filename: str, vev_data: np.array, header: dict[str, int]
) -> None:
    answer = _read_correlators_binary(
        corr_file, filename, vev_file=create_file(header, vev_data)
    )
    assert (
        (answer.vevs[VEV_INDEXING_COLUMNS] == columns_from_header(header, vev=True))
        .all()
        .all()
    )


def test_read_correlators_binary_preserves_data_in_vev(
    corr_file: BinaryIO, filename: str, header: dict[str, int], vev_data: np.array
) -> None:
    answer = _read_correlators_binary(
        corr_file, filename, vev_file=create_file(header, vev_data)
    )
    assert (answer.vevs["glue_bins"] == vev_data).all()


#### Correlators


def test_read_correlators_binary_has_correct_columns(
    corr_file: BinaryIO, filename: str
) -> None:
    answer = _read_correlators_binary(corr_file, filename)
    assert (answer.correlators.columns == CORRELATOR_COLUMNS).all()


def test_read_correlators_binary_has_indexing_columns_consistent_with_header(
    corr_file: BinaryIO, filename: str, header: dict[str, int]
) -> None:
    answer = _read_correlators_binary(corr_file, filename)
    assert (
        (answer.correlators[CORRELATOR_INDEXING_COLUMNS] == columns_from_header(header))
        .all()
        .all()
    )


def test_read_correlators_binary_preserves_data(
    corr_file: BinaryIO, filename: str, data: np.array
) -> None:
    answer = _read_correlators_binary(corr_file, filename)
    assert (answer.correlators["glue_bins"] == data).all()
