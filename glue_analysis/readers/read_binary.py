#!/usr/bin/env python3
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

from ..correlator import CorrelatorEnsemble

LENGTH_OF_CORRELATOR_INDEXING = {
    "MC_Time": lambda header: header["Nbin"],
    "Blocking_index": lambda header: header["Nbl"],
    "Internal": lambda header: header["Nop"],
    "Time": lambda header: int(header["LT"] / 2 + 1),
}
CORRELATOR_INDEXING_COLUMNS = [
    "MC_Time",
    "Blocking_index2",
    "Internal2",
    "Blocking_index1",
    "Internal1",
    "Time",
]
NUMBERS = "0123456789"
CORRELATOR_COLUMNS = CORRELATOR_INDEXING_COLUMNS + ["glue_bins"]
VEV_INDEXING_COLUMNS = ["MC_Time", "Blocking_index", "Internal"]
VEV_COLUMNS = VEV_INDEXING_COLUMNS + ["Vac_exp"]
HEADER_NAMES = ["LX", "LY", "LZ", "LT", "Nc", "Nbin", "bin_size", "Nop", "Nbl"]
SIZE_OF_FLOAT = 8
HEADER_LENGTH = len(HEADER_NAMES) * SIZE_OF_FLOAT


class ParsingError(Exception):
    pass


def read_correlators_binary(
    corr_filename: str,
    channel: str = "",
    vev_filename: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> CorrelatorEnsemble:
    with open(corr_filename, "rb") as corr_file:
        if vev_filename:
            with open(vev_filename, "rb") as vev_file:
                return _read_correlators_binary(
                    corr_file, corr_filename, channel, vev_file, metadata
                )

        return _read_correlators_binary(
            corr_file, corr_filename, channel, None, metadata
        )


def _read_correlators_binary(
    corr_file: BinaryIO,
    filename: str,
    channel: str = "",
    vev_file: BinaryIO | None = None,
    metadata: dict[str, Any] | None = None,
) -> CorrelatorEnsemble:
    correlators = CorrelatorEnsemble(filename)
    correlators.metadata = _assemble_metadata(corr_file, metadata)
    correlators.correlators = _read(corr_file, correlators.metadata, vev=False)
    if vev_file:
        correlators.vevs = _read(vev_file, correlators.metadata, vev=True)
    correlators._frozen = True
    return correlators


def _read(file: BinaryIO, header: dict[str, int], vev: bool) -> pd.DataFrame:
    file.seek(HEADER_LENGTH)
    correlators = _columns_from_header(header, vev)
    correlators[VEV_COLUMNS[-1] if vev else CORRELATOR_COLUMNS[-1]] = (
        # Should be np.fromfile but workaround for https://github.com/numpy/numpy/issues/2230
        np.frombuffer(file.read(), dtype=np.float64)
    )
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
            raise ParsingError(
                "Metadata contains the following entries which differ from"
                f"the header: {conflicts}."
            )
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


def _columns_from_header(header: dict[str, int], vev: bool) -> pd.DataFrame:
    columns = VEV_INDEXING_COLUMNS if vev else CORRELATOR_INDEXING_COLUMNS
    return (
        pd.MultiIndex.from_product(
            [
                range(
                    1, LENGTH_OF_CORRELATOR_INDEXING[column.strip(NUMBERS)](header) + 1
                )
                for column in columns
            ],
            names=columns,
        )
        .to_frame()
        .reset_index(drop=True)
    )
