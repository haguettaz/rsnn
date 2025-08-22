import logging
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import polars as pl

from .constants import REFRACTORY_PERIOD
from .log import setup_logging

logger = setup_logging(__name__, console_level="INFO", file_level="INFO")


# def kernel(
#     f_times: npt.NDArray[np.float64],
#     times: npt.NDArray[np.float64],
#     eps: float = REFRACTORY_PERIOD / 2,
# ):
#     """
#     Evaluate the kernel with respect to the firing times, at the prescribed times.
#     The complexity is linear in times, and logarithmic in f_times.

#     Args:
#         f_times (_type_): _description_
#         times (_type_): _description_
#     """
#     right_ids = np.clip(
#         np.searchsorted(f_times, times, side="left"), 1, f_times.size - 1
#     )
#     return (
#         1.0
#         - np.minimum(
#             np.abs(f_times[right_ids] - times), np.abs(f_times[right_ids - 1] - times)
#         ).clip(max=eps)
#         / eps
#     )


# def compute_precision(
#     ref_f_times: List[npt.NDArray[np.float64]],
#     f_times: List[npt.NDArray[np.float64]],
#     a_times: npt.NDArray[np.float64],
#     n_channels: int,
# ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
#     """
#     Compute the precision of the adjusted firing times with respect to the reference.

#     Args:
#         ref_f_times (list of np.ndarray): Sorted reference spike times.
#         f_times (list of np.ndarray): Sorted spike times of the spike train to evaluate.
#         a_times (np.ndarray): Array of times adjustment to evaluate the precision at.
#         n_channels (int): Number of channels.

#     Returns:
#         np.ndarray: Precision values for each time adjustment in a_times.
#         np.ndarray: Adjusted times.
#     """

#     precision = np.zeros_like(a_times)

#     for f_times_c, r_f_times_c in zip(f_times, ref_f_times):
#         if f_times_c.size > 0:
#             precision += (
#                 kernel(r_f_times_c, f_times_c[:, None] + a_times[None, :])
#             ).sum(axis=0) / f_times_c.size

#         else:
#             precision += np.ones_like(a_times)

#     return precision / n_channels, a_times


# def compute_recall(
#     ref_f_times: List[npt.NDArray[np.float64]],
#     f_times: List[npt.NDArray[np.float64]],
#     a_times: npt.NDArray[np.float64],
#     n_channels: int,
# ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
#     """
#     Compute the recall of the adjusted firing times with respect to the reference.

#     Args:
#         ref_f_times (list of np.ndarray): Sorted reference spike times.
#         f_times (list of np.ndarray): Sorted spike times of the spike train to evaluate.
#         a_times (np.ndarray): Array of times adjustment to evaluate the precision at.
#         n_channels (int): Number of channels.

#     Returns:
#         np.ndarray: recall values for each time adjustment in a_times.
#         np.ndarray: Adjusted times.
#     """

#     recall = np.zeros_like(a_times)

#     for f_times_c, r_f_times_c in zip(f_times, ref_f_times):
#         if r_f_times_c.size > 0:
#             recall += (kernel(f_times_c, r_f_times_c[:, None] - a_times[None, :])).sum(
#                 axis=0
#             ) / r_f_times_c.size

#         else:
#             recall += np.ones_like(a_times)

#     return recall / n_channels, a_times


def compute_best_recall(spikes, spikes_ref, n_neurons, eps):
    adjustment = spikes.join(spikes_ref, on="neuron").select(
        (pl.col("time_ref") - pl.col("time")).unique().alias("delta_adj")
    )

    # Recall
    n_silent_neurons_ref = n_neurons - spikes_ref.select("neuron").n_unique()
    spikes_ref_adj = spikes_ref.join(adjustment, how="cross").with_columns(
        (pl.col("time_ref") - pl.col("delta_adj")).alias("time_ref_adj")
    )
    dist_adj = spikes_ref_adj.sort("time_ref_adj").join_asof(
        spikes.sort("time"),
        by="neuron",
        left_on="time_ref_adj",
        right_on="time",
        strategy="nearest",
        tolerance=eps,
        check_sortedness=False,
        coalesce=False,
    )
    rec_adj = (
        dist_adj.group_by("delta_adj", "neuron")
        .agg(
            (1 - ((pl.col("time_ref_adj") - pl.col("time")).abs() / eps))
            .fill_null(0.0)
            .mean()
            .alias("measure")
        )
        .group_by("delta_adj")
        .agg(pl.col("measure").sum())
    )

    rec_best = (
        n_silent_neurons_ref + rec_adj.select(pl.col("measure").max()).item()
    ) / n_neurons

    return rec_best


def compute_best_precision(spikes, spikes_ref, n_neurons, eps):
    adjustment = spikes.join(spikes_ref, on="neuron").select(
        (pl.col("time_ref") - pl.col("time")).unique().alias("delta_adj")
    )

    # Precision
    n_silent_neurons = n_neurons - spikes.select("neuron").n_unique()
    spikes_adj = spikes.join(adjustment, how="cross").with_columns(
        (pl.col("time") + pl.col("delta_adj")).alias("time_adj")
    )
    dist_adj = spikes_adj.sort("time_adj").join_asof(
        spikes_ref.sort("time_ref"),
        by="neuron",
        left_on="time_adj",
        right_on="time_ref",
        strategy="nearest",
        tolerance=eps,
        check_sortedness=False,
        coalesce=False,
    )
    prec_adj = (
        dist_adj.group_by("delta_adj", "neuron")
        .agg(
            (1 - ((pl.col("time_adj") - pl.col("time_ref")).abs() / eps))
            .fill_null(0.0)
            .mean()
            .alias("measure")
        )
        .group_by("delta_adj")
        .agg(pl.col("measure").sum())
    )

    prec_best = (
        n_silent_neurons + prec_adj.select(pl.col("measure").max()).item()
    ) / n_neurons

    return prec_best


def compute_best_precision(
    ref_f_times: List[npt.NDArray[np.float64]],
    f_times: List[npt.NDArray[np.float64]],
    n_channels: int,
    period: Optional[float | np.float64] = None,
) -> Tuple[float, float]:
    """
    Compute the precision of the best adjusted firing times with respect to the reference.

    Args:
        ref_f_times (list of np.ndarray): Sorted reference spike times.
        f_times (list of np.ndarray): Sorted spike times of the spike train to evaluate.
        n_channels (int): Number of channels.
        period (Optional[float | np.float64]): Period for periodic evaluation. If None, no periodicity is considered.

    Returns:
        tuple: Best precision and corresponding adjustment.
    """
    if period is None:
        a_times = np.unique(
            np.concatenate(
                [
                    (r_f_times_c[None, :] - f_times_c[:, None]).reshape(-1)
                    for f_times_c, r_f_times_c in zip(f_times, ref_f_times)
                ]
                + [np.zeros(1)]
            )
        )
        f_times = [np.sort(f_times_c) for f_times_c in f_times]
        ref_f_times = [np.sort(r_f_times_c) for r_f_times_c in ref_f_times]
        precision, a_times = compute_precision(
            ref_f_times, f_times, a_times, n_channels
        )
        best_id = np.argmax(precision)
        return precision[best_id], a_times[best_id]

    else:
        a_times = np.unique(
            np.concatenate(
                [
                    (r_f_times_c[None, :] - f_times_c[:, None]).reshape(-1) % period
                    for f_times_c, r_f_times_c in zip(f_times, ref_f_times)
                ]
                + [np.zeros(1)]
            )
        )
        # a_times_mod = np.unique(a_times % period)
        f_times = [np.sort(f_times_c % period) for f_times_c in f_times]
        ref_f_times = [np.sort(r_f_times_c % period) for r_f_times_c in ref_f_times]
        ext_r_f_times = [
            np.concatenate(
                [
                    np.array([r_f_times_c[-1] - period]),
                    r_f_times_c,
                    r_f_times_c + period,
                    np.array([r_f_times_c[0] + 2 * period]),
                ]
            ).reshape(-1)
            for r_f_times_c in ref_f_times
        ]

        precision, a_times = compute_precision(
            ext_r_f_times, f_times, a_times, n_channels
        )
        best_id = np.argmax(precision)
        return precision[best_id], a_times[best_id]


def compute_best_recall(
    ref_f_times: List[npt.NDArray[np.float64]],
    f_times: List[npt.NDArray[np.float64]],
    n_channels: int,
    period: Optional[float | np.float64] = None,
):
    """
    Compute the recall of the best adjusted firing times with respect to the reference.

    Args:
        ref_f_times (list of np.ndarray): Sorted reference spike times.
        f_times (list of np.ndarray): Sorted spike times of the spike train to evaluate.
        n_channels (int): Number of channels.
        period (Optional[float | np.float64]): Period for periodic evaluation. If None, no periodicity is considered.

    Returns:
        tuple: Best recall and corresponding adjustment.
    """
    if period is None:
        a_times = np.unique(
            np.concatenate(
                [
                    (r_f_times_c[None, :] - f_times_c[:, None]).reshape(-1)
                    for f_times_c, r_f_times_c in zip(f_times, ref_f_times)
                ]
                + [np.zeros(1)]
            )
        )
        f_times = [np.sort(f_times_c) for f_times_c in f_times]
        ref_f_times = [np.sort(r_f_times_c) for r_f_times_c in ref_f_times]
        recall, a_times = compute_recall(ref_f_times, f_times, a_times, n_channels)
        best_id = np.argmax(recall)
        return recall[best_id], a_times[best_id]

    else:
        a_times = np.unique(
            np.concatenate(
                [
                    (r_f_times_c[None, :] - f_times_c[:, None]).reshape(-1) % period
                    for f_times_c, r_f_times_c in zip(f_times, ref_f_times)
                ]
                + [np.zeros(1)]
            )
        )

        f_times = [np.sort(f_times_c % period) for f_times_c in f_times]
        ref_f_times = [np.sort(r_f_times_c % period) for r_f_times_c in ref_f_times]
        ext_f_times = [
            np.concatenate(
                [
                    np.array([f_times_c[-1] - 2 * period]),
                    f_times_c - period,
                    f_times_c,
                    np.array([f_times_c[0] + period]),
                ]
            ).reshape(-1)
            for f_times_c in f_times
        ]

        recall, a_times = compute_recall(ref_f_times, ext_f_times, a_times, n_channels)
        best_id = np.argmax(recall)
        return recall[best_id], a_times[best_id]
