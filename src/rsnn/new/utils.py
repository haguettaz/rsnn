from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from numba import njit
from scipy.special import lambertw

from .log import setup_logging

logger = setup_logging(__name__, console_level="INFO", file_level="INFO")


def first_crossing(
    start,
    length,
    c0,
    c1,
    f_thresh,
) -> np.ndarray:
    """
    Calculate the first (ascending) crossing time of the states defined by the coefficients `c0` and `c1` with the threshold `f_thresh`, in their respective time intervals.
    If a crossing does not occur within the interval, NaN is returned for that state.

    Parameters
    ----------
    start : float64 ndarray of shape (n_states,)
        Start times for each state.
    length : float64 ndarray of shape (n_states,)
        Lengths of the states for each state.
    c0 : float64 ndarray of shape (n_states,)
        First-order coefficients for each state.
    c1 : float64 ndarray of shape (n_states,)
        Second-order coefficients for each state.
    f_thresh : float64 ndarray of shape (n_states,)
        Firing threshold for each state.

    Returns
    -------
    float64 np.ndarray
        First crossing times for each state.

    Warnings
    --------
    The states arrays must be sorted by start times for correct results.
    """

    dt0 = -(lambertw(-f_thresh / c1 * np.exp(-c0 / c1), 0)) - c0 / c1
    dt1 = -(lambertw(-f_thresh / c1 * np.exp(-c0 / c1), -1)) - c0 / c1
    dt2 = np.log(c0 / f_thresh)

    new_f_times = np.where(
        c0 < f_thresh,
        np.where(
            c1 > 0,
            np.where(
                np.isreal(dt0) & (dt0 >= 0.0) & (dt0 < length), start + dt0.real, np.nan
            ),
            np.where(
                c1 < 0,
                np.where(
                    np.isreal(dt1) & (dt1 >= 0.0) & (dt1 < length),
                    start + dt1.real,
                    np.nan,
                ),
                np.where((dt2 >= 0.0) & (dt2 < length), start + dt2, np.nan),
            ),
        ),
        start,
    )
    return new_f_times


def find_maximum_violation(
    c0: npt.NDArray[np.float64],
    c1: npt.NDArray[np.float64],
    length: npt.NDArray[np.float64],
    lim: npt.NDArray[np.float64],
) -> Optional[Tuple[np.float64, np.intp, np.float64]]:
    """
    Find the maximum violation of constraint condition across intervals.

    Computes the maximum violation of the condition:
    c0[n] + c1[n] * dt * exp(-dt) > lim[n] for 0 <= dt < length[n]

    Parameters
    ----------
    c0 : ndarray of float64
        Zero-order coefficients for each interval.
    c1 : ndarray of float64
        First-order coefficients for each interval.
    length : ndarray of float64
        Length of each interval.
    lim : ndarray of float64
        Maximum allowed value for each interval.

    Returns
    -------
    violation_info : tuple of (float64, intp, float64) or None
        If violation exists, returns (max_violation, interval_index, dt).
        If no violation, returns None.
        - max_violation : Maximum violation value found
        - interval_index : Index of interval where maximum violation occurs
        - dt : Time offset within interval where violation occurs
    """
    dt = np.vstack(
        [np.zeros_like(c0), np.clip(1 - c0 / c1, 0.0, length), length]
    )  # shape (3, n_intervals)
    dv = np.clip(
        np.nan_to_num(c0[np.newaxis, :] + c1[np.newaxis, :] * dt) * np.exp(-dt)
        - lim[np.newaxis, :],
        0.0,
        None,
    )  # shape (3, n_intervals)
    imax = np.unravel_index(
        np.argmax(dv), dv.shape
    )  # index tuple of the maximum value in v

    if dv[imax] > 0.0:
        return dv[imax], imax[1], dt[imax]

    return None


def modulo_with_offset(
    x: npt.NDArray[np.float64],
    period: float | np.float64 | npt.NDArray[np.float64],
    offset: float | np.float64 | npt.NDArray[np.float64] = 0.0,
) -> npt.NDArray[np.float64]:
    """
    Apply modulo operation with custom offset range.

    Computes modulo operation such that results lie in [offset, offset + period).

    Parameters
    ----------
    x : ndarray of float64
        Input array to apply modulo operation.
    period : float64 or ndarray of float64
        Period for the modulo operation.
    offset : float64 or ndarray of float64, optional
        Starting point of the modulo range. Default is 0.0.

    Returns
    -------
    result : ndarray of float64
        Array with values mapped to [offset, offset + period).

    Examples
    --------
    >>> x = np.array([1.5, 3.7, -0.8])
    >>> modulo_with_offset(x, period=2.0, offset=-1.0)
    array([-0.5, -0.3,  0.2])
    """

    return x - period * np.floor_divide(x - offset, period)


@njit
def fscan_states(
    start: npt.NDArray[np.float64],
    ic0: npt.NDArray[np.float64],
    ic1: npt.NDArray[np.float64],
    c0: npt.NDArray[np.float64],
    c1: npt.NDArray[np.float64],
):
    """
    Forward scan states updating cumulative coefficients over time.

    Updates cumulative coefficients c0 and c1 by recursively incorporating
    independent coefficients ic0 and ic1 with exponential decay based on
    time intervals.

    Parameters
    ----------
    start : ndarray of float64, shape (n_states, n_others) or (n_states, 1)
        Start times for each state.
    ic0 : ndarray of float64, shape (n_states, n_others)
        Independent zero-order coefficients for recursive updates.
    ic1 : ndarray of float64, shape (n_states, n_others)
        Independent first-order coefficients for recursive updates.
    c0 : ndarray of float64, shape (n_states, n_others)
        Cumulative zero-order coefficients. Modified in-place.
    c1 : ndarray of float64, shape (n_states, n_others)
        Cumulative first-order coefficients. Modified in-place.

    Warnings
    --------
    The states arrays must be sorted by start times for correct results.

    Notes
    -----
    This function is JIT-compiled with numba for performance.
    The update formula applies exponential decay: exp(-length) where
    length = start[n] - start[n-1].
    """
    c0[0] = ic0[0]
    c1[0] = ic1[0]
    for n in range(1, start.shape[0]):
        length = start[n] - start[n - 1]
        exp_fading = np.exp(-length)
        c0[n] = ic0[n] + (c0[n - 1] + c1[n - 1] * length) * exp_fading
        c1[n] = ic1[n] + c1[n - 1] * exp_fading


def find_last_f_times(f_times, f_sources, f_sources_counts):
    """
    Find the last firing time for each unique source neuron.

    Given arrays of firing times and source neurons, returns the most recent
    firing time for each unique source neuron.

    Parameters
    ----------
    f_times : array_like
        Array of firing times corresponding to each source.
    f_sources : array_like
        Array of source neuron identifiers.

    Returns
    -------
    last_times : ndarray
        Last firing time for each unique source neuron.
    unique_sources : ndarray
        Unique source neuron identifiers corresponding to last_times.

    Notes
    -----
    Uses lexicographic sorting to efficiently find the last occurrence
    of each source neuron.
    """
    unique_sources = np.unique(f_sources)
    sorter = np.lexsort((f_times, f_sources))
    indices_last = (
        np.searchsorted(f_sources, unique_sources, side="right", sorter=sorter) - 1
    )
    return f_times[sorter][indices_last], f_sources[sorter][indices_last]


def neurons_counts_in_sources(
    sources: npt.NDArray[np.int64],
    neurons: npt.NDArray[np.int64],
    # counts: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """
    Count occurrences of each neuron in the sorted sources array.

    Efficiently counts how many times each neuron appears in the sources
    array using binary search.

    Parameters
    ----------
    sources : ndarray of int64
        Sorted array of source neuron identifiers.
    neurons : ndarray of int64
        Array of neuron identifiers to count.

    Returns
    -------
    counts : ndarray of int64
        Number of occurrences of each neuron in sources array.

    Warnings
    --------
    The sources array must be sorted for correct results.

    Examples
    --------
    >>> sources = np.array([0, 0, 1, 1, 1, 2])
    >>> neurons = np.array([0, 1, 2, 3])
    >>> neurons_counts_in_sources(sources, neurons)
    array([2, 3, 1, 0])
    """
    return np.searchsorted(sources, neurons, side="right") - np.searchsorted(
        sources, neurons, side="left"
    )


def receive_in_spikes(
    f_times,
    f_sources,
    f_sources_counts,
    in_sources,
    in_delays,
    in_others,
    in_sources_counts,
):
    """
    Process incoming spikes at the neuron level with delays.

    Computes the arrival times and associated data for incoming spikes
    by adding transmission delays to firing times and replicating data
    according to connectivity patterns.

    Parameters
    ----------
    f_times : array_like
        Firing times of source neurons.
    f_sources : array_like
        Source neuron identifiers for each firing time.
    f_sources_counts : array_like
        Number of firings produced by each source neuron.
    in_sources : array_like
        Source neuron identifiers for incoming connections.
    in_delays : array_like
        Transmission delays for each incoming connection.
    in_others : array_like
        Additional data (e.g., weights) for each incoming connection.
    in_sources_counts : array_like
        Number of outgoing connections for each source neuron.

    Returns
    -------
    in_times : ndarray
        Arrival times of incoming spikes (firing_time + delay).
    in_others_expanded : ndarray
        Expanded additional data corresponding to each incoming spike.

    Notes
    -----
    This function operates at the neuron level, handling the expansion
    and timing calculations for spike propagation through the network.
    """
    in_times = np.repeat(f_times, in_sources_counts[f_sources]) + np.repeat(
        in_delays, f_sources_counts[in_sources]
    )
    in_others = np.repeat(in_others, f_sources_counts[in_sources])
    return in_times, in_others
