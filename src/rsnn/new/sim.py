from typing import Callable, Optional, Tuple

import numpy as np
import numpy.typing as npt
from numba import njit, prange
from scipy.sparse.csgraph import floyd_warshall

from .constants import FIRING_THRESHOLD, REFRACTORY_RESET
from .utils import *

N_STATES = 10000  # Maximum number of states per neuron


def run(
    n_neurons: int,
    f_times: npt.NDArray[np.float64],
    f_sources: npt.NDArray[np.int64],
    f_thresh: npt.NDArray[np.float64],
    in_sources: npt.NDArray[np.int64],
    in_delays: npt.NDArray[np.float64],
    in_weights: npt.NDArray[np.float64],
    tmin: float,
    tmax: float,
    f_thresh_noise: Optional[Callable] = None,
):
    """Run the simulation.

    Warning: This function assumes that the synaptic inputs are sorted by source neuron.

    Args:
        tmin (_type_): _description_
        tmax (_type_): _description_
        in_sources (_type_): _description_
        in_delays (_type_): _description_
        in_weights (_type_): _description_
        f_times (_type_): _description_
        f_sources (_type_): _description_
        f_thresh (_type_): _description_
        f_thresh_noise (callable, optional): _description_. Defaults to None.
    """
    neurons = np.arange(n_neurons)

    # Sort initial firing times and sources
    f_sorter = np.lexsort((f_times, f_sources))
    f_sources = np.take_along_axis(f_sources, f_sorter, axis=-1)
    f_times = np.take_along_axis(f_times, f_sorter, axis=-1)

    # Initialize the neuron counts for initial spikes
    f_neurons_counts = neurons_counts_in_sources(f_sources, neurons)

    # Sort the inputs per source neuron and delays, and update corresponding neuron counts
    in_sorter = np.lexsort((in_delays, in_sources))
    in_sources = np.take_along_axis(in_sources, in_sorter, axis=-1)
    in_delays = np.take_along_axis(in_delays, in_sorter, axis=-1)
    in_weights = np.take_along_axis(in_weights, in_sorter, axis=-1)

    # Initialize the neuron counts for in_sources
    in_neurons_counts = np.empty((n_neurons, n_neurons), dtype=np.int64)
    for l in range(n_neurons):
        in_neurons_counts[l] = neurons_counts_in_sources(in_sources[l], neurons)

    # Initialize propagation delay matrix (i.e., the minimum delay between each pair of neurons)
    min_delays = init_min_delays(in_sources, in_delays)
    floyd_warshall(min_delays, overwrite=True)

    # Create the initial states
    start, c0, c1, ic0, ic1 = create_states(
        n_neurons,
        f_times,
        f_sources,
        f_neurons_counts,
        in_sources,
        in_delays,
        in_weights,
        in_neurons_counts,
    )

    if f_thresh_noise is None:
        f_thresh_noise = lambda size: np.full(size, FIRING_THRESHOLD)

    while tmin < tmax:
        # 1. Clean states before tmin
        clean_states(tmin, start, c0, c1, ic0, ic1)

        # 2. Sort states (only the removed states have to be placed at the end)
        sorter = np.argsort(start, axis=0)
        start = np.take_along_axis(start, sorter, axis=0)
        c0 = np.take_along_axis(c0, sorter, axis=0)
        c1 = np.take_along_axis(c1, sorter, axis=0)
        ic0 = np.take_along_axis(ic0, sorter, axis=0)
        ic1 = np.take_along_axis(ic1, sorter, axis=0)

        # 3. Calculate next firing times
        new_f_times, new_f_sources = next_f_times(
            f_thresh,
            start,
            c0,
            c1,
            ic0,
            ic1,
            min_delays,
        )

        # 4. Fire spikes and reset states
        fire_spikes(
            new_f_times,
            new_f_sources,
            f_times,
            f_sources,
            f_thresh,
            start,
            c0,
            c1,
            ic0,
            ic1,
            f_thresh_noise,
        )

        # 5.a. Sort new firing times and sources
        new_f_sorter = np.argsort(new_f_sources, axis=0)
        new_f_sources = np.take_along_axis(new_f_sources, new_f_sorter, axis=0)
        new_f_times = np.take_along_axis(new_f_times, new_f_sorter, axis=0)
        f_neurons_counts = neurons_counts_in_sources(new_f_sources, neurons)

        # 5.b. Propagate spikes through the network
        propagate_spikes(
            new_f_times,
            new_f_sources,
            f_neurons_counts,
            in_sources,
            in_delays,
            in_weights,
            in_neurons_counts,
            start,
            ic0,
            ic1,
        )

        # 6. Sort states
        sorter = np.argsort(start, axis=0)
        start = np.take_along_axis(start, sorter, axis=0)
        c0 = np.take_along_axis(c0, sorter, axis=0)
        c1 = np.take_along_axis(c1, sorter, axis=0)
        ic0 = np.take_along_axis(ic0, sorter, axis=0)
        ic1 = np.take_along_axis(ic1, sorter, axis=0)

        tmin = np.nanmin(
            new_f_times, initial=np.inf
        )  # if no firings, tmin is set to inf


def create_states(
    neurons,
    f_times,
    f_sources,
    f_neurons_counts,
    in_sources,
    in_delays,
    in_weights,
    in_neurons_counts,
):
    n_neurons = neurons.size

    # Initialize the states
    start = np.full((N_STATES, n_neurons), np.nan, dtype=np.float64)
    start[0, :] = -np.inf  # initial state
    start[1, :] = np.inf  # final state
    c0 = np.zeros_like(start, dtype=np.float64)
    c1 = np.zeros_like(start, dtype=np.float64)
    ic0 = np.zeros_like(start, dtype=np.float64)
    ic1 = np.zeros_like(start, dtype=np.float64)

    # Propagate initial spikes into states
    propagate_spikes(
        f_times,
        f_sources,
        f_neurons_counts,
        in_sources,
        in_delays,
        in_weights,
        in_neurons_counts,
        start,
        ic0,
        ic1,
    )

    # Flush the input spikes that arrived before the last firing time and add refractory reset
    f_indices = np.nonzero(f_neurons_counts) # indices of neurons with at least one firing
    f_last_indices = np.concatenate(
        (np.cumsum(f_neurons_counts), np.array([f_sources.size - 1]))
    ) - 1
    last_f_sources = f_sources[f_last_indices[f_indices]]
    last_f_times = f_times[f_last_indices[f_indices]]
    reset_states(
        last_f_times, last_f_sources, start, ic0, ic1,
    )

    return start, c0, c1, ic0, ic1


@njit(parallel=True)
def init_min_delays(in_sources, in_delays):
    """
    Compute the minimum direct connection delay between all neuron pairs.

    For each postsynaptic neuron `j`, this function finds all presynaptic neurons `i` listed in `in_sources[j]` and records the smallest delay from `i` to `j` in the output matrix.
    If no direct connection exists, the delay remains `np.inf`.

    Parameters
    ----------
    in_sources : ndarray of int64, shape (n_neurons, n_inputs)
        For each postsynaptic neuron (rows), the source neuron IDs of its incoming
        synapses.
        Must be sorted by source ID, and for equal IDs sorted by delay.
    in_delays : ndarray of float64, shape (n_neurons, n_inputs)
        The synaptic delays corresponding to `in_sources`.

    Returns
    -------
    min_delays : ndarray of float64, shape (n_neurons, n_neurons)
        Matrix where entry `(i, j)` is the minimum delay from presynaptic neuron `i` to postsynaptic neuron `j`.
        Entries are `np.inf` if no direct connection exists.
    """

    n_neurons = in_sources.shape[0]
    neurons = np.arange(n_neurons)
    min_delays = np.full((n_neurons, n_neurons), np.inf, dtype=np.float64)
    for l in prange(n_neurons):
        pos = np.searchsorted(in_sources[l], neurons, side="left")
        ids = np.argwhere(in_sources[l, pos] == neurons).flatten()
        min_delays[ids, l] = in_delays[l, pos[ids]]

    return min_delays


@njit(parallel=True)
def reset_states(f_times, f_sources, start, ic0, ic1):
    """Reset the states of the neurons that fire.


    Args:
        f_times (npt.NDArray[np.float64]): The firing times for the neurons.
        f_sources (npt.NDArray[np.int64]): The source neurons for the firings.
    """

    for n in prange(f_sources.size):
        l, ft = f_sources[n], f_times[n]
        ipos = np.searchsorted(start[:, l], ft, side="left")
        start[1:ipos, l] = np.nan
        start[-1, l] = ft
        ic0[-1, l] = REFRACTORY_RESET
        ic1[-1, l] = 0.0


def clean_states(tmin, start, c0, c1, ic0, ic1):
    """
    Consolidate all states occurring before `tmin` into the last state before `tmin`, and remove (mark as NaN) earlier states.

    This function is designed for per-neuron state histories that are:
    - **Sorted** along the first axis by their start time (`start`).
    - **Framed** by an initial state at `-np.inf` (index 0) and a final state at `np.inf` (right before the trailing NaNs).

    For each neuron:
      1. Find the index of the most recent state before `tmin`.
      2. Accumulate contributions from all previous states into that state's coefficients (`c0`, `c1`).
      3. Mark all earlier states (rows 1 to that index) as removed by setting their `start` entries to `NaN`.

    Parameters
    ----------
    tmin : float
        The cutoff time. Only the last state before this time is preserved.
    start : float64 ndarray of shape (n_states, n_neurons)
        Start times for each state (must be sorted along axis 0, with NaNs trailing).
    c0 : float64 ndarray of shape (n_states, n_neurons)
        Zero-order cumulative coefficients (will be updated in place).
    c1 : float64 ndarray of shape (n_states, n_neurons)
        First-order cumulative coefficients (will be updated in place).
    ic0 : float64 ndarray of shape (n_states, n_neurons)
        Zero-order independent coefficients (used for recursive updates).
    ic1 : float64 ndarray of shape (n_states, n_neurons)
        First-order independent coefficients (used for recursive updates).
    """
    n_neurons = start.shape[1]
    for l in range(n_neurons):
        idx = (
            np.searchsorted(start[:, l], tmin, side="right") - 1
        )  # always >= 0 since -np.inf is at the start

        length = start[idx, l] - start[1:idx, l]
        exp_decay = np.exp(-length)
        c0[idx, l] = ic0[idx, l] + np.sum(
            (ic0[1:idx, l] + ic1[1:idx, l] * length) * exp_decay, axis=0
        )
        c1[idx, l] = ic1[idx, l] + np.sum(ic1[1:idx, l] * exp_decay)

        start[1:idx, l] = np.nan
        ic0[1:idx, l] = 0.0
        ic1[1:idx, l] = 0.0
        c0[1:idx, l] = 0.0
        c1[1:idx, l] = 0.0


def next_f_times(f_thresh, start, c0, c1, ic0, ic1, min_delays):
    """
    Calculate the next (causally independent) firings produced in the network, among all neurons.

    Warning: This function requires the states to be sorted by start.

    Returns:
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]: A tuple containing:
            - f_times: Array of firing times.
            - f_sources: Array of firing sources (neuron indices).
    """
    n_intervals, n_neurons = start.shape
    f_times = np.full(n_neurons, np.nan, dtype=np.float64)
    max_f_times = np.full(n_neurons, np.inf, dtype=np.float64)
    n_indices = np.arange(n_neurons)
    nf_indices = n_indices[:]

    for n in range(1, n_intervals - 1):  # start
        # early exit if no neuron can possibly fire
        if nf_indices.size == 0:
            break

        # update the coefficients
        prev_length = start[n, nf_indices] - start[n - 1, nf_indices]
        c0[n, nf_indices] = (
            c0[n - 1, nf_indices] + np.nan_to_num(prev_length) * c1[n - 1, nf_indices]
        ) * np.exp(-(prev_length)) + ic0[n, nf_indices]
        c1[n, nf_indices] = (
            c1[n - 1, nf_indices] * np.exp(-(prev_length)) + ic1[n, nf_indices]
        )

        # update the candidate firing times
        f_times[nf_indices] = first_crossing(
            start[n, nf_indices],
            np.clip(
                start[n + 1, nf_indices],
                None,
                max_f_times[nf_indices],
            )
            - start[n, nf_indices],
            c0[n, nf_indices],
            c1[n, nf_indices],
            f_thresh[nf_indices],
        )

        # update the maximum firing times
        max_f_times = np.nanmin(
            np.nan_to_num(f_times[None, :], nan=np.inf)
            + min_delays,  # can be further optimized... use Numba???
            initial=np.inf,
            axis=1,
        )

        # update the indices of the neurons that can possibly fire
        nf_indices = n_indices[np.isnan(f_times) & (start[n] < max_f_times)]

    f_sources = np.argwhere(f_times <= max_f_times).flatten()
    f_times = f_times[f_sources]
    return f_times, f_sources


def fire_spikes(
    new_f_times,
    new_f_sources,
    f_times,
    f_sources,
    f_thresh,
    start,
    c0,
    c1,
    ic0,
    ic1,
    f_thresh_noise: Callable,
):
    # Concatenate the new firing times and sources with the existing ones
    f_times = np.concatenate((f_times, new_f_times))
    f_sources = np.concatenate((f_sources, new_f_sources))

    # Flush the input spikes and reset to the refractory state
    reset_states(
        new_f_times, new_f_sources, start, ic0, ic1
    )

    # Apply threshold noise
    f_thresh[f_sources] = f_thresh_noise(f_sources.size)

    return f_times, f_sources, start, c0, c1, ic0, ic1


def propagate_spikes(
    f_times,
    f_sources,
    f_sources_counts,
    in_sources,
    in_weights,
    in_delays,
    in_sources_counts,
    start,
    ic0,
    ic1,
):
    """at the network level"""
    n_neurons, _ = in_sources.shape

    for l in range(n_neurons):
        in_times, in_weights = receive_in_spikes(
            f_times,
            f_sources,
            f_sources_counts[l],
            in_sources[l],
            in_weights[l],
            in_delays[l],
            in_sources_counts[l],
        )

        start[-in_times.size :, l] = in_times
        ic0[-in_times.size :, l] = np.zeros_like(in_weights)
        ic1[-in_times.size :, l] = in_weights
