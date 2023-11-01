#!/usr/bin/env python3

import logging

import numpy as np
import pyerrors as pe


class CorrelatorEnsemble:
    """
    Represents a full ensemble of gluonic correlation functions.
    """

    _frozen = False

    def __init__(self, filename):
        self.correlators = []
        self.vevs = []
        self.metadata = {}
        self.filename = filename

    @property
    def NT(self):
        return max(self.correlators.Time)

    @property
    def num_ops(self):
        return max(self.correlators.Op_index1)

    @property
    def num_bins(self):
        return max(self.correlators.Bin_index)

    @property
    def has_consistent_vevs(self):
        if max(self.vevs.Op_index) != self.num_ops:
            logging.warning("Wrong number of operators in vevs")
            return False
        if len(set(self.vevs.Op_index)) != self.num_ops:
            logging.warning("Missing operators in vevs")

        if max(self.vevs.Bin_index) != self.num_bins:
            logging.warning("Wrong number of bins in vevs")
            return False
        if len(set(self.vevs.Bin_index)) != self.num_bins:
            logging.warning("Missing bins in vevs")
            return False

        for op_idx in range(1, self.num_ops + 1):
            for bin_idx in range(1, self.num_bins + 1):
                if (
                    sum(
                        (self.vevs.Op_index == op_idx)
                        & (self.vevs.Bin_index == bin_idx)
                    )
                    != 1
                ):
                    logging.warning(f"Missing {op_idx=}, {bin_idx=} in vevs")
                    return False

        return True

    @property
    def is_consistent(self):
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

        if len(set(self.correlators.Bin_index)) != self.num_bins:
            logging.warning("Missing bins")
            return False

        if len(self.correlators) != self.num_bins * self.NT * self.num_ops**2:
            logging.warning("Total length not consistent")
            return False

        if self.vevs is not None and not self.has_consistent_vevs:
            return False

        return True

    def get_numpy(self):
        if not self.is_consistent:
            raise ValueError("Data are inconsistent.")
        sorted_correlators = self.correlators.sort_values(
            by=["Bin_index", "Time", "Op_index1", "Op_index2"]
        )
        return sorted_correlators.Correlation.values.reshape(
            self.num_bins, self.NT, self.num_ops, self.num_ops
        )

    def get_numpy_vevs(self):
        if not self.is_consistent:
            raise ValueError("Data are inconsistent")
        sorted_vevs = self.vevs.sort_values(by=["Bin_index", "Op_index"])
        return sorted_vevs.Vac_exp.values.reshape(self.num_bins, self.num_ops)

    def get_pyerrors(self, subtract=False):
        if subtract and (self.vevs is None):
            raise ValueError("Can't subtract vevs that have not been read.")

        array = self.get_numpy()
        if subtract:
            vevs = self.get_numpy_vevs()
            vev_matrix = vevs[:, :, np.newaxis] * vevs[:, np.newaxis, :] / self.NT**2
        else:
            vev_matrix = np.zeros((self.num_bins, self.num_ops, self.num_ops))
            # array -= vev_matrix * vev_matrix.swapaxes(2, 3) / self.NT ** 2

        correlation_covariances = np.asarray(
            [
                [
                    [
                        pe.Obs([array[:, t_idx, op_idx1, op_idx2]], ["glue_bins"])
                        - pe.Obs([vev_matrix[:, op_idx1, op_idx2]], ["glue_bins"])
                        for op_idx2 in range(self.num_ops)
                    ]
                    for op_idx1 in range(self.num_ops)
                ]
                for t_idx in range(self.NT)
            ]
        )

        return pe.Corr(correlation_covariances)
