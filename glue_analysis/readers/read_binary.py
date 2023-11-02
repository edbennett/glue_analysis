#!/usr/bin/env python3
from typing import Any, BinaryIO

import numpy as np

from ..correlator import CorrelatorEnsemble

HEADER_NAMES = ["LX", "LY", "LZ", "LT", "Nc", "Nbin", "bin_size", "Nop", "Nbl"]
SIZE_OF_FLOAT = 8


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
    correlators.metadata = _read_header(corr_file)
    if metadata:
        if conflicting_keys := [key for key in metadata if key in HEADER_NAMES]:
            raise ParsingError(
                f"Metadata contains the keys {conflicting_keys} "
                "which are supposed to be read from the header."
            )
        correlators.metadata |= metadata
    correlators._frozen = True
    return correlators


def _read_header(corr_file: BinaryIO) -> dict[str, int]:
    return {
        name: int(val)
        for name, val in zip(
            HEADER_NAMES,
            np.frombuffer(
                corr_file.read(len(HEADER_NAMES) * SIZE_OF_FLOAT), dtype=np.float64
            ),
            strict=True,
        )
    }
