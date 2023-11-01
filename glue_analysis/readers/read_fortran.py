#!/usr/bin/env python3

from functools import lru_cache

import pandas as pd

from ..correlator import CorrelatorEnsemble

@lru_cache(maxsize=8)
def read_correlators_fortran(filename, channel="", vev_filename=None, metadata=None):
    correlators = CorrelatorEnsemble(filename)
    correlators.correlators = pd.read_csv(filename, delim_whitespace=True)
    correlators.correlators["channel"] = channel
    if vev_filename:
        correlators.vevs = pd.read_csv(vev_filename, delim_whitespace=True)
        correlators.vevs["channel"] = channel

    if not metadata:
        metadata = {}

    correlators.metadata = metadata
    correlators._frozen = True

    return correlators
