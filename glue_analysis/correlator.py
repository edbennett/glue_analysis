#!/usr/bin/env python3

import logging
from collections.abc import Callable
from typing import Any, Self

import numpy as np
import pandas as pd
import pandera as pa
import pyerrors as pe
from pandera.typing import DataFrame as DataFrameType

_DESCRIPTIONS = {
    #
    "MC_Time": "Index enumerating the Monte Carlo samples.",
    #
    "Time": "Physical euclidean time coordinate "
    "along which correlation is measured.",
    #
    "Internal": "Any further internal structure, e.g.,"
    "an index enumerating interpolating operators, "
    "a blocking or smearing level, "
    "or any combination thereof.",
    #
    "Correlation": "Measured values of the correlators.",
    #
    "Vac_exp": "Measured values of the vacuum expectation values (VEVs).",
    #
    "Check_Internals_equal": "Internal1 and Internal2 are supposed to form"
    "square matrix, so they must be identical up to reordering.",
    #
    "Check_unique_indexing": "The index columns are supposed "
    "to make for a unique index.",
}
CorrelatorData = pa.DataFrameSchema(
    {
        "MC_Time": pa.Column(int, required=True, description=_DESCRIPTIONS["MC_Time"]),
        "Time": pa.Column(int, required=True, description=_DESCRIPTIONS["Time"]),
        "Internal1": pa.Column(required=True, description=_DESCRIPTIONS["Internal"]),
        "Internal2": pa.Column(required=True, description=_DESCRIPTIONS["Internal"]),
        "Correlation": pa.Column(
            float, required=True, description=_DESCRIPTIONS["Correlation"]
        ),
    },
    checks=[
        pa.Check(
            lambda df: (
                df["Internal1"].sort_values().values
                == df["Internal2"].sort_values().values
            ).all(),
            description=_DESCRIPTIONS["Check_Internals_equal"],
        ),
        pa.Check(
            lambda df: not df[["MC_Time", "Time", "Internal1", "Internal2"]]
            .duplicated()
            .any(),
            description=_DESCRIPTIONS["Check_unique_indexing"],
        ),
    ],
)
VEVData = pa.DataFrameSchema(
    {
        "MC_Time": pa.Column(int, required=True, description=_DESCRIPTIONS["MC_Time"]),
        "Internal": pa.Column(required=True, description=_DESCRIPTIONS["Internal"]),
        "Vac_exp": pa.Column(
            float, required=True, description=_DESCRIPTIONS["Vac_exp"]
        ),
    }
)


class FrozenError(Exception):
    pass


def only_on_consistent_data(func: Callable) -> Callable:
    # Completely generic function, ignore typing here
    def func_with_check(self, *args, **kwargs):  # noqa: ANN202,ANN001,ANN002,ANN003
        if not self.is_consistent:
            raise ValueError("Data are inconsistent.")
        return func(self, *args, **kwargs)

    return func_with_check


class CorrelatorEnsemble:
    """
    Represents a full ensemble of gluonic correlation functions.
    """

    filename: str
    _correlators: DataFrameType[CorrelatorData]
    _vevs: DataFrameType[VEVData]
    metadata: dict[str, Any]
    ensemble_name: str
    _frozen: bool = False

    def __init__(self: Self, filename: str, ensemble_name: str | None = None) -> None:
        self.filename = filename
        self.ensemble_name = ensemble_name if ensemble_name else "glue_bins"

    def freeze(self: Self) -> Self:
        if not isinstance(self._correlators, pd.DataFrame):
            raise TypeError(
                "Correlator data is expected to be pandas.Dataframe "
                f"but {type(self._correlators)} was found."
            )

        if hasattr(self, "_vevs") and not isinstance(self._vevs, pd.DataFrame):
            raise TypeError(
                "VEV data is expected to be pandas.Dataframe "
                f"but {type(self._vevs)} was found."
            )

        CorrelatorData.validate(self._correlators)
        if hasattr(self, "_vevs"):
            VEVData.validate(self._vevs)
        self._frozen = True
        return self

    @property
    def correlators(self: Self) -> DataFrameType[CorrelatorData]:
        return self._correlators

    @correlators.setter
    def correlators(self: Self, value: Any) -> None:  # noqa: ANN401
        if not self.frozen:
            self._correlators = value
        else:
            raise FrozenError(
                "This instance is frozen. "
                "You are not allowed to modify correlators anymore."
            )

    @property
    def vevs(self: Self) -> DataFrameType[CorrelatorData]:
        if hasattr(self, "_vevs"):
            return self._vevs
        raise AttributeError("Vevs is not set for this instance.")

    @vevs.setter
    def vevs(self: Self, value: Any) -> None:  # noqa: ANN401
        if not self.frozen:
            self._vevs = value
        else:
            raise FrozenError(
                "This instance is frozen. "
                "You are not allowed to modify vevs anymore."
            )

    @property
    def frozen(self: Self) -> bool:
        return self._frozen

    @property
    def NT(self: Self) -> int:
        return max(self._correlators.Time)

    @property
    def num_internal(self: Self) -> int:
        return max(self._correlators.Internal1)

    @property
    def num_samples(self: Self) -> int:
        return max(self._correlators.MC_Time)

    @property
    def has_consistent_vevs(self: Self) -> bool:
        if max(self.vevs.Internal) != self.num_internal:
            logging.warning("Wrong number of internal dof in vevs")
            return False
        if len(set(self.vevs.Internal)) != self.num_internal:
            logging.warning("Missing internal dof in vevs")

        if max(self.vevs.MC_Time) != self.num_samples:
            logging.warning("Wrong number of samples in vevs")
            return False
        if len(set(self.vevs.MC_Time)) != self.num_samples:
            logging.warning("Missing samples in vevs")
            return False

        for internal in range(1, self.num_internal + 1):
            for sample in range(1, self.num_samples + 1):
                if (
                    sum(
                        (self.vevs.Internal == internal) & (self.vevs.MC_Time == sample)
                    )
                    != 1
                ):
                    logging.warning(f"Missing {internal=}, {sample=} in vevs")
                    return False

        return True

    @property
    def is_consistent(self: Self) -> bool:
        if not self._frozen:
            raise ValueError("Data must be frozen to check consistency.")
        if max(self._correlators.Internal2) != self.num_internal:
            logging.warning("Inconsistent numbers of internal")
            return False
        if set(self._correlators.Internal2) != set(self._correlators.Internal1):
            logging.warning("Inconsistent internal dof pairings")
            return False
        if len(set(self._correlators.Internal1)) != self.num_internal:
            logging.warning("Internal1 missing one or more internal")
            return False
        if len(set(self._correlators.Internal2)) != self.num_internal:
            logging.warning("Internal2 missing one or more internal")
            return False

        if len(set(self._correlators.Time)) != self.NT:
            logging.warning("Missing time slices")
            return False

        if len(set(self._correlators.MC_Time)) != self.num_samples:
            logging.warning("Missing samples")
            return False

        if len(self._correlators) != self.num_samples * self.NT * self.num_internal**2:
            logging.warning("Total length not consistent")
            return False

        if hasattr(self, "_vevs") and not self.has_consistent_vevs:
            return False

        return True

    @only_on_consistent_data
    def get_numpy(self: Self) -> np.array:
        sorted_correlators = self._correlators.sort_values(
            by=["MC_Time", "Time", "Internal1", "Internal2"]
        )
        return sorted_correlators.Correlation.values.reshape(
            self.num_samples, self.NT, self.num_internal, self.num_internal
        )

    @only_on_consistent_data
    def get_numpy_vevs(self: Self) -> np.array:
        sorted_vevs = self._vevs.sort_values(by=["MC_Time", "Internal"])
        return sorted_vevs.Vac_exp.values.reshape(self.num_samples, self.num_internal)

    def get_pyerrors(self: Self, subtract: bool = False) -> pe.Corr:
        if subtract and not hasattr(self, "_vevs"):
            raise ValueError("Can't subtract vevs that have not been read.")

        return pe.Corr(
            to_obs_array(self.get_numpy(), self.ensemble_name)
            - (
                np.outer(
                    *(2 * [to_obs_array(self.get_numpy_vevs(), self.ensemble_name)])
                )
                / self.NT**2
                if subtract
                else 0.0
            )
        )


def to_obs_array(array: np.array, ensemble_name: str) -> pe.Obs:
    if array.ndim == 1:
        return pe.Obs([array], [ensemble_name])

    return np.asarray(
        [
            to_obs_array(sub_array, ensemble_name)
            for sub_array in np.moveaxis(array, 1, 0)
        ]
    )
