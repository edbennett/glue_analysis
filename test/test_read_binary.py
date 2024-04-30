#!/usr/bin/env python3
from io import BytesIO
from typing import BinaryIO

import numpy as np
import pandas as pd
import pytest

from glue_analysis.correlator import CorrelatorData, VEVData
from glue_analysis.readers.read_binary import (
    CORRELATOR_INDEXING_COLUMNS,
    HEADER_NAMES,
    VEV_INDEXING_COLUMNS,
    ParsingError,
    _read_correlator_binary,
)


@pytest.fixture()
def filename() -> str:
    return "testname.txt"


@pytest.fixture()
def header() -> dict[str, int]:
    return {name: i + 1 for i, name in enumerate(HEADER_NAMES)}


OFFSET_FOR_1_INDEXING = 1


def _correlator_length(length_in_time: int) -> int:
    return int(length_in_time / 2 + 1)


def index_from_header(header: dict[str, int], *, vev: bool = False) -> pd.MultiIndex:
    index_ranges = [
        range(OFFSET_FOR_1_INDEXING, header["Nbin"] + OFFSET_FOR_1_INDEXING),
        range(
            OFFSET_FOR_1_INDEXING, header["Nop"] * header["Nbl"] + OFFSET_FOR_1_INDEXING
        ),
    ]
    if not vev:
        index_ranges += [
            range(
                OFFSET_FOR_1_INDEXING,
                header["Nop"] * header["Nbl"] + OFFSET_FOR_1_INDEXING,
            ),
            range(
                OFFSET_FOR_1_INDEXING,
                _correlator_length(header["LT"]) + OFFSET_FOR_1_INDEXING,
            ),
        ]
    return pd.MultiIndex.from_product(
        index_ranges,
        names=VEV_INDEXING_COLUMNS if vev else CORRELATOR_INDEXING_COLUMNS,
    )


def create_data(header: dict[str, int], *, vev: bool = False) -> np.array:
    return np.random.default_rng().random(index_from_header(header, vev=vev).shape[0])


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
    answer = _read_correlator_binary(corr_file, filename)
    assert answer.filename == filename


def test_read_correlators_binary_does_not_create_vev_if_not_given(
    corr_file: BinaryIO, filename: str
) -> None:
    answer = _read_correlator_binary(corr_file, filename)
    assert not hasattr(answer, "vevs")


def test_read_correlators_binary_freezes_the_ensemble(
    corr_file: BinaryIO, filename: str
) -> None:
    answer = _read_correlator_binary(corr_file, filename)
    assert answer.frozen


### Actually functional behavior

#### Metadata


def test_read_correlators_binary_makes_metadata_from_header_constant(
    filename: str,
) -> None:
    header = {name: 1 for name in HEADER_NAMES}
    corr_file = create_file(header, create_data(header))
    answer = _read_correlator_binary(corr_file, filename)
    assert answer.metadata == header


def test_read_correlators_binary_makes_metadata_from_header_rising(
    filename: str,
) -> None:
    header = {name: i for i, name in enumerate(HEADER_NAMES)}
    corr_file = create_file(header, create_data(header))
    answer = _read_correlator_binary(corr_file, filename)
    assert answer.metadata == header


def test_read_correlators_binary_merges_header_with_metadata(
    corr_file: BinaryIO, filename: str, header: dict[str, int]
) -> None:
    metadata = {"some": "metadata"}
    answer = _read_correlator_binary(corr_file, filename, metadata=metadata)
    assert answer.metadata == header | metadata


def test_read_correlators_binary_raises_on_conflicting_metadata(
    corr_file: BinaryIO, filename: str
) -> None:
    metadata = {HEADER_NAMES[0]: "conflict with header info"}
    with pytest.raises(ParsingError):
        _read_correlator_binary(corr_file, filename, metadata=metadata)


def test_read_correlators_binary_accepts_consistently_doubly_specified_metadata(
    corr_file: BinaryIO, filename: str, header: dict[str, int]
) -> None:
    metadata = {HEADER_NAMES[0]: header[HEADER_NAMES[0]]}
    answer = _read_correlator_binary(corr_file, filename, metadata=metadata)
    assert answer.metadata == header


#### VEVs


def test_read_correlators_binary_has_correct_columns_in_vev(
    corr_file: BinaryIO, filename: str, vev_file: BinaryIO
) -> None:
    answer = _read_correlator_binary(corr_file, filename, vev_file=vev_file)
    assert set(answer.vevs.columns) == set(
        VEVData.get_metadata()[None]["columns"].keys()
    )


def test_read_correlators_binary_has_indexing_columns_consistent_with_header_in_vev(
    corr_file: BinaryIO, filename: str, vev_data: np.array, header: dict[str, int]
) -> None:
    answer = _read_correlator_binary(
        corr_file, filename, vev_file=create_file(header, vev_data)
    )
    assert (answer.vevs.index == index_from_header(header, vev=True)).all(axis=None)


def test_read_correlators_binary_takes_user_specified_mc_time_in_vevs(
    corr_file: BinaryIO, filename: str, vev_data: np.array, header: dict[str, int]
) -> None:
    answer = _read_correlator_binary(
        corr_file,
        filename,
        vev_file=create_file(header, vev_data),
        metadata={"MC_Time": [-(i + 1) for i in range(header["Nbin"])]},
    )
    index = index_from_header(header, vev=True).to_frame()
    index["MC_Time"] *= -1
    assert (answer.vevs.index == pd.MultiIndex.from_frame(index)).all(axis=None)


def test_read_correlators_binary_preserves_data_in_vev(
    corr_file: BinaryIO, filename: str, header: dict[str, int], vev_data: np.array
) -> None:
    answer = _read_correlator_binary(
        corr_file, filename, vev_file=create_file(header, vev_data)
    )
    assert (answer.vevs["Vac_exp"] == vev_data).all()


#### Correlators


def test_read_correlators_binary_has_correct_columns(
    corr_file: BinaryIO, filename: str
) -> None:
    answer = _read_correlator_binary(corr_file, filename)
    assert set(answer.correlators.columns) == set(
        CorrelatorData.get_metadata()[None]["columns"].keys()
    )


def test_read_correlators_binary_has_indexing_columns_consistent_with_header(
    corr_file: BinaryIO, filename: str, header: dict[str, int]
) -> None:
    answer = _read_correlator_binary(corr_file, filename)
    assert (answer.correlators.index == index_from_header(header, vev=False)).all(
        axis=None
    )


def test_read_correlators_binary_takes_user_specified_mc_time(
    corr_file: BinaryIO, filename: str, vev_data: np.array, header: dict[str, int]
) -> None:
    answer = _read_correlator_binary(
        corr_file,
        filename,
        vev_file=create_file(header, vev_data),
        metadata={"MC_Time": [-(i + 1) for i in range(header["Nbin"])]},
    )
    index = index_from_header(header, vev=False).to_frame()
    index["MC_Time"] *= -1
    assert (answer.correlators.index == pd.MultiIndex.from_frame(index)).all(axis=None)


def test_read_correlators_binary_preserves_data(
    corr_file: BinaryIO, filename: str, data: np.array
) -> None:
    answer = _read_correlator_binary(corr_file, filename)
    assert (answer.correlators["Correlation"] == data).all()


def test_read_correlators_binary_raises_on_inconsistent_file(
    corr_file: BinaryIO, filename: str
) -> None:
    corr_file.truncate(104)  # now it's shorter than the header promises
    with pytest.raises(ValueError, match="Inconsistent header"):
        _read_correlator_binary(corr_file, filename)


def corrupt_file(corr_file: BinaryIO) -> None:
    # last number is corrupted (too few bytes to be read as float64) now:
    corr_file.truncate(103)


def test_read_correlators_binary_raises_on_corrupted_data(
    corr_file: BinaryIO, filename: str
) -> None:
    corrupt_file(corr_file)
    with pytest.raises(ValueError, match="Corrupted data"):
        _read_correlator_binary(corr_file, filename)
