#!/usr/bin/env python3
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

from ..correlator import CorrelatorEnsemble

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
    if vev_file:
        correlators.vevs = _read_vevs(vev_file, metadata)
    correlators._frozen = True
    return correlators


def _assemble_metadata(
    corr_file: BinaryIO, metadata: dict[str, Any] | None
) -> dict[str, Any]:
    final_metadata = _read_header(corr_file)
    if metadata:
        if conflicting_keys := [key for key in metadata if key in HEADER_NAMES]:
            raise ParsingError(
                f"Metadata contains the keys {conflicting_keys} "
                "which are supposed to be read from the header."
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


def _read_vevs(vev_file: BinaryIO, metadata: dict[str, Any] | None) -> pd.DataFrame:
    vev_file.seek(HEADER_LENGTH)
    vevs = pd.DataFrame(
        {
            "Bin_index": np.arange(10, dtype=np.float64),
            "Op_index": np.arange(10, dtype=np.float64),
            "Vac_exp": np.frombuffer(vev_file.read(), dtype=np.float64),
        }
    )
    vev_file.seek(0)
    return vevs
