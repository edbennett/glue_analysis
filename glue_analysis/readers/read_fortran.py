#!/usr/bin/env python3

from functools import lru_cache
from logging import warning
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

from glue_analysis.correlator import CorrelatorEnsemble


@lru_cache(maxsize=8)
def read_correlators_fortran(
    corr_filename: str,
    channel: str = "",
    vev_filename: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> CorrelatorEnsemble:  # pragma: no cover
    with Path(corr_filename).open() as corr_file:
        if vev_filename:
            with Path(vev_filename).open() as vev_file:
                return _read_correlators_fortran(
                    corr_file,
                    corr_filename,
                    channel,
                    vev_file,
                    metadata,
                )

        return _read_correlators_fortran(
            corr_file, corr_filename, channel, None, metadata
        )


def _read_correlator_file(corr_file: TextIO) -> pd.DataFrame:
    return pd.read_csv(
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
            "Op1_index": "Internal1",
            "Op2_index": "Internal2",
        },
        axis="columns",
    )


def _read_vev_file(vev_file: TextIO) -> pd.DataFrame:
    return pd.read_csv(
        vev_file,
        delim_whitespace=True,
        converters={"Bin_index": int, "Op_index": int, "Vac_exp": float},
    ).rename(
        {"Bin_index": "MC_Time", "Time": "Time", "Op_index": "Internal"},
        axis="columns",
    )


def _normalise_vevs(vevs: pd.DataFrame, NT: int, num_configs: int) -> None:
    vevs["Vac_exp"] /= (NT * num_configs / len(set(vevs.MC_Time))) ** 0.5


def _check_ensemble_divisibility(num_configs: int, num_samples: int) -> None:
    if num_configs % num_samples != 0:
        message = (
            f"Number of configurations {num_configs} is not divisible by "
            f"number of samples {num_samples}."
        )
        warning(message)


def _read_correlators_fortran(
    corr_file: TextIO,
    filename: str,
    channel: str = "",
    vev_file: TextIO | None = None,
    metadata: dict[str, Any] | None = None,
) -> CorrelatorEnsemble:
    if not metadata:
        metadata = {}

    if vev_file and (missing := {"NT", "num_configs"} - set(metadata.keys())):
        message = f"{missing} must be specified to normalise VEVs correctly."
        raise ValueError(message)

    correlators = CorrelatorEnsemble(filename)
    correlators.correlators = _read_correlator_file(corr_file)

    _check_ensemble_divisibility(
        metadata.get("num_configs", 0), correlators.num_samples
    )

    if vev_file:
        correlators.vevs = _read_vev_file(vev_file)
        correlators.vevs["channel"] = channel
        _normalise_vevs(correlators.vevs, metadata["NT"], metadata["num_configs"])

    correlators.correlators["channel"] = channel
    correlators.metadata = metadata

    return correlators.freeze()
