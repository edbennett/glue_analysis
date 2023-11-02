#!/usr/bin/env python3

from functools import lru_cache
from typing import Any, TextIO

import pandas as pd

from ..correlator import CorrelatorEnsemble


@lru_cache(maxsize=8)
def read_correlators_fortran(
    corr_filename: str,
    channel: str = "",
    vev_filename: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> CorrelatorEnsemble:
    with open(corr_filename) as corr_file:
        if vev_filename:
            with open(vev_filename) as vev_file:
                return _read_correlators_fortran(
                    corr_file, corr_filename, channel, vev_file, metadata
                )

        return _read_correlators_fortran(
            corr_file, corr_filename, channel, None, metadata
        )


def _read_correlators_fortran(
    corr_file: TextIO,
    filename: str,
    channel: str = "",
    vev_file: TextIO | None = None,
    metadata: dict[str, Any] | None = None,
) -> CorrelatorEnsemble:
    correlators = CorrelatorEnsemble(filename)
    correlators.correlators = pd.read_csv(corr_file, delim_whitespace=True)
    correlators.correlators["channel"] = channel
    if vev_file:
        correlators.vevs = pd.read_csv(vev_file, delim_whitespace=True)
        correlators.vevs["channel"] = channel

    if not metadata:
        metadata = {}

    correlators.metadata = metadata
    correlators._frozen = True

    return correlators
