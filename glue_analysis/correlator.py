#!/usr/bin/env python3

import logging
from collections.abc import Callable
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

    @property
    def NT(self: Self) -> int:
        return max(self.correlators.Time)

    @property
    def num_ops(self: Self) -> int:
        return max(self.correlators.Op_index1)

    @property
    def num_bins(self: Self) -> int:
        return max(self.correlators.MC_Time)

    @property
    def has_consistent_vevs(self: Self) -> bool:
        if max(self.vevs.Op_index) != self.num_ops:
            logging.warning("Wrong number of operators in vevs")
            return False
        if len(set(self.vevs.Op_index)) != self.num_ops:
            logging.warning("Missing operators in vevs")

        if max(self.vevs.MC_Time) != self.num_bins:
            logging.warning("Wrong number of bins in vevs")
            return False
        if len(set(self.vevs.MC_Time)) != self.num_bins:
            logging.warning("Missing bins in vevs")
            return False

        for op_idx in range(1, self.num_ops + 1):
            for bin_idx in range(1, self.num_bins + 1):
                if (
                    sum((self.vevs.Op_index == op_idx) & (self.vevs.MC_Time == bin_idx))
                    != 1
                ):
                    logging.warning(f"Missing {op_idx=}, {bin_idx=} in vevs")
                    return False

        return True

    @property
    def is_consistent(self: Self) -> bool:
        if not self._frozen:
            raise ValueError("Data must be frozen to check consistency.")
        if max(self.correlators.Op_index2) != self.num_ops:
            logging.warning("Inconsistent numbers of operators")
            return False
        if set(self.correlators.Op_index2) != set(self.correlators.Op_index1):
            logging.warning("Inconsistent operator pairings")
            return False
        if len(set(self.correlators.Op_index1)) != self.num_ops:
            logging.warning("Op_index1 missing one or more operators")
            return False
        if len(set(self.correlators.Op_index2)) != self.num_ops:
            logging.warning("Op_index2 missing one or more operators")
            return False

        if len(set(self.correlators.Time)) != self.NT:
            logging.warning("Missing time slices")
            return False

        if len(set(self.correlators.MC_Time)) != self.num_bins:
            logging.warning("Missing bins")
            return False

        if len(self.correlators) != self.num_bins * self.NT * self.num_ops**2:
            logging.warning("Total length not consistent")
            return False

        if hasattr(self, "vevs") and not self.has_consistent_vevs:
            return False

        return True

    @only_on_consistent_data
    def get_numpy(self: Self) -> np.array:
        sorted_correlators = self.correlators.sort_values(
            by=["MC_Time", "Time", "Op_index1", "Op_index2"]
        )
        return sorted_correlators.Correlation.values.reshape(
            self.num_bins, self.NT, self.num_ops, self.num_ops
        )

    @only_on_consistent_data
    def get_numpy_vevs(self: Self) -> np.array:
        sorted_vevs = self.vevs.sort_values(by=["MC_Time", "Op_index"])
        return sorted_vevs.Vac_exp.values.reshape(self.num_bins, self.num_ops)

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
