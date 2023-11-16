#!/usr/bin/env python3

from functools import lru_cache
from logging import warn
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

from glue_analysis.correlator import CorrelatorEnsemble


@lru_cache(maxsize=8)
def read_correlators_fortran(
    corr_filename: str,
    NT: int,
    num_configs: int,
    channel: str = "",
    vev_filename: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> CorrelatorEnsemble:  # pragma: no cover
    with Path(corr_filename).open() as corr_file:
        if vev_filename:
            with Path(vev_filename).open() as vev_file:
                return _read_correlators_fortran(
                    corr_file, NT, num_configs, corr_filename, channel, vev_file, metadata
                )

        return _read_correlators_fortran(
            corr_file, NT, num_configs, corr_filename, channel, None, metadata
        )


def _read_correlators_fortran(
    corr_file: TextIO,
    NT: int,
    num_configs: int,
    filename: str,
    channel: str = "",
    vev_file: TextIO | None = None,
    metadata: dict[str, Any] | None = None,
) -> CorrelatorEnsemble:
    correlators = CorrelatorEnsemble(filename)
    correlators.correlators = pd.read_csv(
        corr_file,
        delim_whitespace=True,
        converters={
            "Bin_index": int,
            "Time": int,
            "Op_index1": int,
            "Op_index2": int,
            "Correlation": float,
        },
    ).rename(
        {
            "Bin_index": "MC_Time",
            "Time": "Time",
            "Op_index1": "Internal1",
            "Op_index2": "Internal2",
        },
        axis="columns",
    )
    correlators.correlators["channel"] = channel
    if vev_file:
        correlators.vevs = pd.read_csv(
            vev_file,
            delim_whitespace=True,
            converters={"Bin_index": int, "Op_index": int, "Vac_exp": float},
        ).rename(
            {"Bin_index": "MC_Time", "Time": "Time", "Op_index": "Internal"},
            axis="columns",
        )
        correlators.vevs["channel"] = channel
        correlators.vevs["Vac_exp"] /= (NT * num_configs / correlators.num_samples) ** 0.5

    if not metadata:
        metadata = {}

    correlators.metadata = metadata

    if num_configs % correlators.num_samples != 0:
        warn(
            f"Number of configurations {num_configs} is not divisible by "
            f"number of samples {correlators.num_samples}."
        )

    return correlators.freeze()
