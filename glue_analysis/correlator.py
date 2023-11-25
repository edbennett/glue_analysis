#!/usr/bin/env python3

from typing import Any, Self

import numpy as np
import pandas as pd
import pandera as pa
import pyerrors as pe
from pandera.typing import DataFrame as DataFrameType

_COLUMN_DESCRIPTIONS = {
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
}
_CHECK_DESCRIPTIONS = {
    #
    "Check_Internals_equal": "Internal1 and Internal2 are supposed to form"
    "square matrix, so they must be identical up to reordering.",
}
CorrelatorData = pa.DataFrameSchema(
    {
        "Correlation": pa.Column(
            float, required=True, description=_COLUMN_DESCRIPTIONS["Correlation"]
        ),
    },
    index=pa.MultiIndex(
        [
            pa.Index(int, description=_COLUMN_DESCRIPTIONS["MC_Time"], name="MC_Time"),
            pa.Index(
                int, description=_COLUMN_DESCRIPTIONS["Internal"], name="Internal1"
            ),
            pa.Index(
                int, description=_COLUMN_DESCRIPTIONS["Internal"], name="Internal2"
            ),
            pa.Index(int, description=_COLUMN_DESCRIPTIONS["Time"], name="Time"),
        ],
        strict=True,
        ordered=False,
        unique=["MC_Time", "Internal1", "Internal2", "Time"],
    ),
    checks=[
        pa.Check(
            lambda df: (
                df.index.get_level_values("Internal1").sort_values()
                == df.index.get_level_values("Internal2").sort_values()
            ).all(),
            description=_CHECK_DESCRIPTIONS["Check_Internals_equal"],
            name="Check_Internals_equal",
        ),
    ],
)
VEVData = pa.DataFrameSchema(
    {
        "Vac_exp": pa.Column(
            float, required=True, description=_COLUMN_DESCRIPTIONS["Vac_exp"]
        ),
    },
    index=pa.MultiIndex(
        [
            pa.Index(int, description=_COLUMN_DESCRIPTIONS["MC_Time"], name="MC_Time"),
            pa.Index(
                int, description=_COLUMN_DESCRIPTIONS["Internal"], name="Internal"
            ),
        ],
        strict=True,
        ordered=False,
        unique=["MC_Time", "Internal"],
    ),
)


class FrozenError(Exception):
    pass


class DataInconsistencyError(Exception):
    pass


def cross_validate(
    corr: DataFrameType[CorrelatorData], vevs: DataFrameType[VEVData]
) -> None:
    vevs = vevs.index.to_frame(index=False)
    if not (
        # It's sufficient to check this one way round (and not with Internal1/2
        # interchanged) because their consistency is assured from other checks.
        corr.index.to_frame(index=False)
        .groupby(by=["Internal2", "Time"])
        .apply(
            lambda df: sorted(df[["MC_Time", "Internal1"]].to_numpy().tolist())
            == sorted(vevs[["MC_Time", "Internal"]].to_numpy().tolist())
        )
    ).all():
        message = (
            "VEVs and correlators have differing MC_Time and Internal axes. "
            "Are they coming from the same ensemble?"
        )
        raise DataInconsistencyError(message)


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
            message = (
                "Correlator data is expected to be pandas.Dataframe "
                f"but {type(self._correlators)} was found."
            )
            raise TypeError(message)

        if hasattr(self, "_vevs") and not isinstance(self._vevs, pd.DataFrame):
            message = (
                "VEV data is expected to be pandas.Dataframe "
                f"but {type(self._vevs)} was found."
            )
            raise TypeError(message)

        message = (
            "Non-unique index, "
            "should be pa.errors.SchemaError "
            "but fails due to some incompatibility."
        )
        try:
            CorrelatorData.validate(self._correlators)
        except ValueError as ex:
            if "Columns with duplicate values are not supported in stack" in str(ex):
                # see https://github.com/unionai-oss/pandera/issues/1328
                raise ValueError(message) from ex
            raise
        if hasattr(self, "_vevs"):
            try:
                VEVData.validate(self._vevs)
            except ValueError as ex:
                if "Columns with duplicate values are not supported in stack" in str(
                    ex
                ):
                    # see https://github.com/unionai-oss/pandera/issues/1328
                    raise ValueError(message) from ex
                raise
            cross_validate(self._correlators, self._vevs)
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
            message = (
                "This instance is frozen. "
                "You are not allowed to modify correlators anymore."
            )
            raise FrozenError(message)

    @property
    def vevs(self: Self) -> DataFrameType[CorrelatorData]:
        if hasattr(self, "_vevs"):
            return self._vevs
        message = "Vevs is not set for this instance."
        raise AttributeError(message)

    @vevs.setter
    def vevs(self: Self, value: Any) -> None:  # noqa: ANN401
        if not self.frozen:
            self._vevs = value
        else:
            message = (
                "This instance is frozen. "
                "You are not allowed to modify vevs anymore."
            )
            raise FrozenError(message)

    @property
    def frozen(self: Self) -> bool:
        return self._frozen

    @property
    def num_timeslices(self: Self) -> int:
        return len(self._correlators.index.unique("Time"))

    @property
    def num_internal(self: Self) -> int:
        return len(self._correlators.index.unique("Internal1"))

    @property
    def num_samples(self: Self) -> int:
        return len(self._correlators.index.unique("MC_Time"))

    def get_numpy(self: Self) -> np.array:
        sorted_correlators = self._correlators.sort_values(
            by=["MC_Time", "Time", "Internal1", "Internal2"]
        )
        return sorted_correlators.Correlation.to_numpy().reshape(
            self.num_samples, self.num_timeslices, self.num_internal, self.num_internal
        )

    def get_numpy_vevs(self: Self) -> np.array:
        sorted_vevs = self._vevs.sort_values(by=["MC_Time", "Internal"])
        return sorted_vevs.Vac_exp.to_numpy().reshape(
            self.num_samples, self.num_internal
        )

    def get_pyerrors(self: Self, *, subtract: bool = False) -> pe.Corr:
        if subtract and not hasattr(self, "_vevs"):
            message = "Can't subtract vevs that have not been read."
            raise ValueError(message)

        return pe.Corr(
            to_obs_array(self.get_numpy(), self.ensemble_name)
            - (
                np.outer(
                    *(2 * [to_obs_array(self.get_numpy_vevs(), self.ensemble_name)])
                )
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
