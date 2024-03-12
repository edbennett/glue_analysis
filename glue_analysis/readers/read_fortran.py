#!/usr/bin/env python3
from copy import deepcopy
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

from glue_analysis.auxiliary import NoneContext
from glue_analysis.correlator import CorrelatorData, CorrelatorEnsemble, VEVData


def read_correlators_fortran(
    corr_filename: str,
    channel: str = "",
    vev_filename: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> CorrelatorEnsemble:  # pragma: no cover
    with (
        Path(corr_filename).open("r") as corr_file,
        (
            # typechecking fails on @contextmanager
            Path(vev_filename).open("r") if vev_filename else NoneContext()  # type: ignore[attr-defined]
        ) as vev_file,
    ):
        return _read_correlators_fortran(
            corr_file, corr_filename, channel, vev_file, metadata
        )


def _read_single_file(file_to_read: TextIO) -> pd.DataFrame:
    return pd.read_csv(
        file_to_read,
        delim_whitespace=True,
        converters={
            "Bin_index": int,
            "Time": int,
            "Op_index1": int,
            "Op_index2": int,
            "Op_index": int,
            "Correlation": float,
            "Vac_exp": float,
        },
    ).rename(
        {
            "Bin_index": "MC_Time",
            "Time": "Time",
            "Op1_index": "Internal1",
            "Op2_index": "Internal2",
            "Op_index": "Internal",
        },
        axis="columns",
    )


def _normalise_vevs(
    vevs: pd.DataFrame, NT: int, num_configs: int, *, inplace: bool = False
) -> None | pd.DataFrame:
    if not inplace:
        vevs = vevs.copy()

    vevs["Vac_exp"] /= (NT * num_configs / len(vevs.index.unique("MC_Time"))) ** 0.5
    if not inplace:
        return vevs

    return None


def _check_ensemble_divisibility(num_configs: int | None, num_samples: int) -> None:
    if num_configs is not None and num_configs % num_samples != 0:
        message = (
            f"Number of configurations {num_configs} is not divisible by "
            f"number of samples {num_samples}."
        )
        raise ValueError(message)


def _read_correlators_fortran(
    corr_file: TextIO,
    filename: str,
    channel: str = "",
    vev_file: TextIO | None = None,
    metadata: dict[str, Any] | None = None,
) -> CorrelatorEnsemble:
    metadata = deepcopy(metadata) if metadata else {}

    if vev_file and (missing := {"NT", "num_configs"} - set(metadata.keys())):
        message = f"{missing} must be specified to normalise VEVs correctly."
        raise ValueError(message)

    correlators = CorrelatorEnsemble(filename)
    correlators.correlators = _read_single_file(corr_file).set_index(
        list(CorrelatorData.index.get_metadata()[None]["columns"].keys()), append=False
    )
    correlators.correlators["channel"] = channel

    _check_ensemble_divisibility(metadata.get("num_configs"), correlators.num_samples)

    if vev_file:
        correlators.vevs = _read_single_file(vev_file).set_index(
            list(VEVData.index.get_metadata()[None]["columns"].keys()), append=False
        )
        correlators.vevs["channel"] = channel
        _normalise_vevs(
            correlators.vevs,
            metadata["NT"],
            metadata["num_configs"],
            inplace=True,
        )

    correlators.metadata = metadata

    return correlators.freeze()
