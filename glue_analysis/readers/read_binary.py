#!/usr/bin/env python3
import os
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Any, BinaryIO, TypeVar

import numpy as np
import pandas as pd

from glue_analysis.auxiliary import NUMBERS, NoneContext, NoneGenerator, ParsingError
from glue_analysis.correlator import CorrelatorEnsemble, concatenate

LENGTH_OF_CORRELATOR_INDEXING = {
    "MC_Time": lambda header: header["Nbin"],
    "Blocking_index": lambda header: header["Nbl"],
    "Internal": lambda header: header["Nop"] * header["Nbl"],
    "Time": lambda header: int(header["LT"] / 2 + 1),
}
CORRELATOR_INDEXING_COLUMNS = [
    "MC_Time",
    "Internal1",
    "Internal2",
    "Time",
]
CORRELATOR_VALUE_COLUMN_NAME = "Correlation"
VEV_VALUE_COLUMN_NAME = "Vac_exp"
VEV_INDEXING_COLUMNS = ["MC_Time", "Internal"]
HEADER_NAMES = ["LX", "LY", "LZ", "LT", "Nc", "Nbin", "bin_size", "Nop", "Nbl"]
SIZE_OF_FLOAT = 8
HEADER_LENGTH = len(HEADER_NAMES) * SIZE_OF_FLOAT


T = TypeVar("T")


def _handle_types(
    elements: Any,  # noqa: ANN401
    value_types: Any,  # noqa: ANN401
    target_type: T,
    # more precisely this should be two @overload's
    # one taking None and returning Generator[None, None, None]
    # and one taking Any (not None) and returning Iterable[Path]
) -> Iterable[T] | Generator[None, None, None]:  # pragma: no cover
    if elements is None:
        return NoneGenerator()
    if isinstance(elements, value_types):
        elements = target_type(elements)  # type: ignore[operator]
        return [elements]
    return elements


def read_correlators_binary(
    corr_filenames: str | os.PathLike | Iterable[str | os.PathLike],
    vev_filenames: str | os.PathLike | Iterable[str | os.PathLike] | None = None,
    metadata_sets: dict[str, Any] | Iterable[dict[str, Any]] | None = None,
) -> CorrelatorEnsemble:  # pragma: no cover
    return concatenate(
        [
            # the first one is never None
            read_correlator_binary(corr_filename, vev_filename, metadata)  # type: ignore[arg-type]
            for corr_filename, vev_filename, metadata in zip(
                _handle_types(corr_filenames, str | os.PathLike, Path),
                _handle_types(vev_filenames, str | os.PathLike, Path),
                _handle_types(metadata_sets, dict, dict[str, Any]),
                strict=vev_filenames is not None,
            )
        ]
    )


def read_correlator_binary(
    corr_filename: Path,
    vev_filename: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> CorrelatorEnsemble:  # pragma: no cover
    with Path(corr_filename).open("rb") as corr_file, (
        # typechecking fails on @contextmanager
        Path(vev_filename).open("rb") if vev_filename else NoneContext()  # type: ignore[attr-defined]
    ) as vev_file:
        return _read_correlators_binary(
            corr_file, str(corr_filename), vev_file, metadata
        )


def _read_correlators_binary(
    corr_file: BinaryIO,
    filename: str,
    vev_file: BinaryIO | None = None,
    metadata: dict[str, Any] | None = None,
) -> CorrelatorEnsemble:
    correlators = CorrelatorEnsemble(filename)
    correlators.metadata = _assemble_metadata(corr_file, metadata)
    correlators.correlators = _read(
        corr_file,
        _index_from_header(
            correlators.metadata,
            CORRELATOR_INDEXING_COLUMNS,
            correlators.metadata.get("MC_Time", None),
        ),
        CORRELATOR_VALUE_COLUMN_NAME,
    )
    if vev_file:
        correlators.vevs = _read(
            vev_file,
            _index_from_header(
                correlators.metadata,
                VEV_INDEXING_COLUMNS,
                correlators.metadata.get("MC_Time", None),
            ),
            VEV_VALUE_COLUMN_NAME,
        )

    return correlators.freeze(perform_expensive_validation=False)


def _read(
    file: BinaryIO,
    # could be more precise, i.e., only indexing portion of
    # DataFrameType[CorrelatorData | VEVData]:
    index: pd.MultiIndex,
    value_column_name: str,
) -> pd.DataFrame:
    file.seek(HEADER_LENGTH)
    try:
        correlators = pd.DataFrame(
            {
                value_column_name:
                # Should be np.fromfile but workaround for https://github.com/numpy/numpy/issues/2230
                np.frombuffer(file.read(), dtype=np.float64)
            },
            index=index,
        )
    except ValueError as exc:
        if "buffer size must be a multiple of element size" in str(exc):
            message = (
                "Corrupted data: The file has the wrong number of bytes "
                "to be read as header + array of float64."
            )
            raise ValueError(message) from exc
        if "does not match length of index" in str(exc):
            file.seek(HEADER_LENGTH)
            length = np.frombuffer(file.read(), dtype=np.float64).shape[0]
            message = (
                f"Inconsistent header: The file content has length {length} "
                f"but the header suggested that it should be {index.shape[0]}."
            )
            raise ValueError(message) from exc
        raise  # pragma: no cover
    file.seek(0)
    return correlators


def _assemble_metadata(
    corr_file: BinaryIO, metadata: dict[str, Any] | None
) -> dict[str, Any]:
    final_metadata = _read_header(corr_file)
    if metadata:
        if conflicting_keys := [
            key
            for key, value in metadata.items()
            if final_metadata.get(key, value) != value
            # if key not in final_metadata, it returns `value` which equals
            # `value`
            # if key in final_metadata, it return the entry from there which is
            # fine if and only if that ones equal to `value` again
        ]:
            conflicts = {
                key: {"metadata": metadata[key], "header": final_metadata[key]}
                for key in conflicting_keys
            }
            message = (
                "Metadata contains the following entries which differ from"
                f"the header: {conflicts}."
            )
            raise ParsingError(message)
        final_metadata |= metadata
    return final_metadata


def _read_header(corr_file: BinaryIO) -> dict[str, int]:
    header = {
        name: int(val)
        for name, val in zip(
            HEADER_NAMES,
            # Should be np.fromfile but workaround for https://github.com/numpy/numpy/issues/2230
            np.frombuffer(corr_file.read(HEADER_LENGTH), dtype=np.float64),
            strict=True,
        )
    }
    corr_file.seek(0)
    return header


def _index_from_header(
    header: dict[str, int], columns: list[str], mc_time: list[int] | None = None
) -> pd.MultiIndex:
    individual_values: list[Any] = [
        range(1, LENGTH_OF_CORRELATOR_INDEXING[column.strip(NUMBERS)](header) + 1)
        for column in columns
    ]
    if mc_time is not None:
        individual_values[0] = mc_time
    return pd.MultiIndex.from_product(individual_values, names=columns)
