#!/usr/bin/env python3

import logging
from collections.abc import Callable
from copy import copy
from typing import Any, Self

import numpy as np
import pandas as pd
import pyerrors as pe

# for type hints, not really enforced as of now:
CorrelatorData = pd.DataFrame
VEVData = pd.DataFrame


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
    correlators: pd.DataFrame
    vevs: pd.DataFrame
    metadata: dict[str, Any]
    ensemble_name: str
    _frozen: bool = False

    def __init__(self: Self, filename: str, ensemble_name: str | None = None) -> None:
        self.filename = filename
        self.ensemble_name = ensemble_name if ensemble_name else "glue_bins"

    def freeze(self: Self) -> Self:
        if isinstance(self.correlators, str):
            raise ValueError("Correlator data has wrong type.")
        if hasattr(self, "vevs") and isinstance(self.vevs, str):
            raise ValueError("VEV data has wrong type.")

        self._frozen = True
        return copy(self)

    @property
    def frozen(self: Self) -> bool:
        return self._frozen

    @property
    def NT(self: Self) -> int:
        return max(self.correlators.Time)

    @property
    def num_internal(self: Self) -> int:
        return max(self.correlators.Internal1)

    @property
    def num_samples(self: Self) -> int:
        return max(self.correlators.MC_Time)

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
        if max(self.correlators.Internal2) != self.num_internal:
            logging.warning("Inconsistent numbers of internal")
            return False
        if set(self.correlators.Internal2) != set(self.correlators.Internal1):
            logging.warning("Inconsistent internal dof pairings")
            return False
        if len(set(self.correlators.Internal1)) != self.num_internal:
            logging.warning("Internal1 missing one or more internal")
            return False
        if len(set(self.correlators.Internal2)) != self.num_internal:
            logging.warning("Internal2 missing one or more internal")
            return False

        if len(set(self.correlators.Time)) != self.NT:
            logging.warning("Missing time slices")
            return False

        if len(set(self.correlators.MC_Time)) != self.num_samples:
            logging.warning("Missing samples")
            return False

        if len(self.correlators) != self.num_samples * self.NT * self.num_internal**2:
            logging.warning("Total length not consistent")
            return False

        if hasattr(self, "vevs") and not self.has_consistent_vevs:
            return False

        return True

    @only_on_consistent_data
    def get_numpy(self: Self) -> np.array:
        sorted_correlators = self.correlators.sort_values(
            by=["MC_Time", "Time", "Internal1", "Internal2"]
        )
        return sorted_correlators.Correlation.values.reshape(
            self.num_samples, self.NT, self.num_internal, self.num_internal
        )

    @only_on_consistent_data
    def get_numpy_vevs(self: Self) -> np.array:
        sorted_vevs = self.vevs.sort_values(by=["MC_Time", "Internal"])
        return sorted_vevs.Vac_exp.values.reshape(self.num_samples, self.num_internal)

    def get_pyerrors(self: Self, subtract: bool = False) -> pe.Corr:
        if subtract and not hasattr(self, "vevs"):
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
