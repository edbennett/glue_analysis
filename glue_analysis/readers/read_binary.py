#!/usr/bin/env python3
from typing import Any, BinaryIO

from ..correlator import CorrelatorEnsemble


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
    correlators._frozen = True
    return correlators
