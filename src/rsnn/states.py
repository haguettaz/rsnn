import numpy as np
import polars as pl
import polars.selectors as cs
from numba import float64, guvectorize
from scipy.special import lambertw

from .constants import REFRACTORY_RESET


@guvectorize(
    [(float64[:], float64[:], float64[:], float64[:], float64[:])],
    "(n),(n),(n),(n)->(n)",
)
def recursively_update_c0(length, w0, c1, c0, res):
    res[0] = c0[0]
    for n in range(1, len(length)):
        res[n] = w0[n] + (res[n - 1] + c1[n - 1] * length[n - 1]) * np.exp(
            -length[n - 1]
        )


@guvectorize([(float64[:], float64[:], float64[:], float64[:])], "(n),(n),(n)->(n)")
def recursively_update_c1(length, w1, c1, res):
    res[0] = c1[0]
    for n in range(1, len(length)):
        res[n] = w1[n] + res[n - 1] * np.exp(-length[n - 1])


def update_coef(states, over=None):
    # Warning: states should be sorted by start (according to the grouping given by over) before being passed to this function

    states = states.with_columns(
        pl.struct(["length", "w1", "c1"])
        .map_batches(
            lambda batch: recursively_update_c1(
                batch.struct.field("length"),
                batch.struct.field("w1"),
                batch.struct.field("c1"),
            ),  # type: ignore
            return_dtype=pl.Float64,
        )
        .over(over)
        .alias("c1")
    )
    states = states.with_columns(
        pl.struct(["length", "w0", "c0", "c1"])
        .map_batches(
            lambda batch: recursively_update_c0(
                batch.struct.field("length"),
                batch.struct.field("w0"),
                batch.struct.field("c1"),
                batch.struct.field("c0"),
            ),  # type: ignore
            return_dtype=pl.Float64,
        )
        .over(over)
        .alias("c0")
    )

    return states


def update_length(states, over=None, fill_value=None):
    # Warning: states should be sorted by start
    states = states.with_columns(
        pl.col("start")
        .diff()
        .shift(-1, fill_value=fill_value)
        .over(over)
        .alias("length"),
    )
    return states


def compute_maxima(states, vmin, k=1, by=None, right_lim=1e-9):
    states = states.with_columns(
        (
            pl.lit(0.0, pl.Float64).alias("left_delta"),
            (1 - pl.col("c0") / pl.col("c1"))
            .clip(0.0, (pl.col("length") - right_lim).clip(0.0))
            .alias("crit_delta"),
            (pl.col("length") - right_lim).clip(0.0).alias("right_delta"),
        )
    )
    states = states.unpivot(
        index=cs.exclude(cs.ends_with("_delta")), value_name="delta"
    ).drop("variable")
    states = states.with_columns(
        ((pl.col("c0") + pl.col("c1") * pl.col("delta")) * (-pl.col("delta")).exp())
        .fill_nan(0.0)  # Fading memory -> 0 at infinity
        .alias("value")
    )

    states = states.filter(pl.col("value") > vmin)
    if by is None:
        max_violations = states.top_k(k=k, by="value")
    else:
        max_violations = (
            states.group_by(by).agg(pl.all().top_k_by("value", k=k)).explode(cs.list())
        )

    max_violations = max_violations.with_columns(
        (pl.col("start") + pl.col("delta")).alias("time")
    )
    return max_violations


def compute_rising_crossing_times(f_thresh, start, length, c0, c1):
    dt = np.where(
        c0 < f_thresh,
        np.where(
            c1 > 0.0,
            -(lambertw(-f_thresh / c1 * np.exp(-c0 / c1), 0)) - c0 / c1,
            np.where(
                c1 < 0.0,
                -(lambertw(-f_thresh / c1 * np.exp(-c0 / c1), -1)) - c0 / c1,
                np.log(c0 / f_thresh),
            ),
        ),
        0.0,
    )
    dt[(dt < 0.0) | (dt >= length) | np.iscomplex(dt)] = np.nan
    return start + np.real(dt)


# @guvectorize([(float64[:], float64[:], float64[:], float64[:])], "(n),(n),(n)->(n)")
# def recursively_update_c0(length, c1, w0, c0):
#     c0[0] = w0[0]
#     for n in range(1, len(length)):
#         c0[n] = w0[n] + (c0[n - 1] + c1[n - 1] * length[n - 1]) * np.exp(
#             -length[n - 1]
#         )


# @guvectorize([(float64[:], float64[:], float64[:])], "(n),(n)->(n)")
# def recursively_update_c1(length, w1, c1):
#     c1[0] = w1[0]
#     for n in range(1, len(length)):
#         c1[n] = w1[n] + c1[n - 1] * np.exp(-length[n - 1])
