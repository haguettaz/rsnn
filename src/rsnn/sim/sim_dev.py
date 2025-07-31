from typing import Tuple

import numpy as np
import numpy.typing as npt
from numba import njit, prange
from scipy.sparse.csgraph import floyd_warshall
from scipy.special import lambertw

FIRING_THRESHOLD = 1.0
REFRACTORY_RESET = -1.0
N_STATES = 10000  # Maximum number of states per neuron


def run(
    f_times: npt.NDArray[np.float64],
    f_sources: npt.NDArray[np.int64],
    f_thresh: npt.NDArray[np.float64],
    in_sources: npt.NDArray[np.int64],
    in_delays: npt.NDArray[np.float64],
    in_weights: npt.NDArray[np.float64],
    tmin: float,
    tmax: float,
    f_thresh_std: float = 0.0,
):
    """Run the simulation.

    Warning: This function assumes that the synaptic inputs are sorted by source neuron.

    Args:
        tmin (_type_): _description_
        tmax (_type_): _description_
        f_thresh (_type_): _description_
        in_sources (_type_): _description_
        in_delays (_type_): _description_
        in_weights (_type_): _description_
        f_times (_type_): _description_
        f_sources (_type_): _description_
        f_thresh_std (float, optional): _description_. Defaults to 0.0.
    """
    # Sort the inputs per source neuron and delays
    in_sources, in_delays, in_weights = sort_inputs(in_sources, in_delays, in_weights)

    # Initialize propagation delay matrix (i.e., the minimum delay between each pair of neurons)
    min_delays = init_min_delays(in_sources, in_delays)
    min_delays = floyd_warshall(min_delays, overwrite=True)

    # Initialize states
    st_start, st_c0, st_c1, st_ic0, st_ic1 = init_states(
        in_sources, in_delays, in_weights, f_times, f_sources
    )

    while tmin < tmax:
        # 1. Clean states before tmin
        st_start, st_c0, st_c1, st_ic0, st_ic1 = clean_states(
            tmin, st_start, st_c0, st_c1, st_ic0, st_ic1
        )

        # 2. Sort states (only the removed states have to be placed at the end)
        st_start, st_c0, st_c1, st_ic0, st_ic1 = sort_states(
            st_start, st_c0, st_c1, st_ic0, st_ic1
        )

        # 3. Calculate next firing times
        new_f_times, new_f_sources = next_f_times(
            f_thresh,
            st_start,
            st_c0,
            st_c1,
            st_ic0,
            st_ic1,
            min_delays,
        )

        # 4. Fire spikes and reset states
        f_times, f_sources, st_start, st_c0, st_c1, st_ic0, st_ic1 = fire_spikes(
            new_f_times,
            new_f_sources,
            # last_f_times,
            f_thresh,
            f_times,
            f_sources,
            st_start,
            st_c0,
            st_c1,
            st_ic0,
            st_ic1,
            f_thresh_std,
        )

        # 5. Propagate spikes through the network
        st_start, st_c0, st_c1, st_ic0, st_ic1 = propagate_spikes(
            new_f_times,
            new_f_sources,
            in_sources,
            in_delays,
            in_weights,
            st_start,
            st_c0,
            st_c1,
            st_ic0,
            st_ic1,
        )

        # 6. Sort states
        st_start, st_c0, st_c1, st_ic0, st_ic1 = sort_states(
            st_start, st_c0, st_c1, st_ic0, st_ic1
        )

        tmin = np.nanmin(
            new_f_times, initial=np.inf
        )  # if no firings, tmin is set to inf


def init_states(f_times, f_sources, in_sources, in_delays, in_weights):
    n_neurons = in_sources.shape[0]
    st_start = np.full((N_STATES, n_neurons), np.nan, dtype=np.float64)
    st_start[0, :] = -np.inf  # initial state
    st_start[1, :] = np.inf  # final state
    st_c0 = np.zeros_like(st_start, dtype=np.float64)
    st_c1 = np.zeros_like(st_start, dtype=np.float64)
    st_ic0 = np.zeros_like(st_start, dtype=np.float64)
    st_ic1 = np.zeros_like(st_start, dtype=np.float64)

    st_start, st_c0, st_c1, st_ic0, st_ic1 = propagate_spikes(
        f_times,
        f_sources,
        in_sources,
        in_delays,
        in_weights,
        st_start,
        st_c0,
        st_c1,
        st_ic0,
        st_ic1,
    )

    # Determine the last firing times for each source neuron
    last_f_times, last_f_souces = find_last_f_times(f_times, f_sources)

    # Flush the input spikes and reset to the refractory state
    st_start, st_c0, st_c1, st_ic0, st_ic1 = reset_states(
        last_f_times, last_f_souces, st_start, st_c0, st_c1, st_ic0, st_ic1
    )

    return st_start, st_c0, st_c1, st_ic0, st_ic1


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


def sort_inputs(
    in_sources: npt.NDArray[np.int64],
    in_delays: npt.NDArray[np.float64],
    in_weights: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Sort each neuron's inputs first by source ID, then by delay.

    Parameters
    ----------
    in_sources : ndarray of int64, shape (n_neurons, n_inputs)
        Source neuron IDs for each postsynaptic neuron.
    in_delays : ndarray of float64, shape (n_neurons, n_inputs)
        Synaptic delays corresponding to `in_sources`.
    in_weights : ndarray of float64, shape (n_neurons, n_inputs)
        Synaptic weights corresponding to `in_sources`.

    Returns
    -------
    in_sources : ndarray of int64
        Source IDs sorted per neuron by (source, delay).
    in_delays : ndarray of float64
        Delays reordered to match the sorted sources.
    in_weights : ndarray of float64
        Weights reordered to match the sorted sources.
    """

    sorter = np.lexsort((in_delays, in_sources), axis=1)
    in_sources = np.take_along_axis(in_sources, sorter, axis=1)
    in_delays = np.take_along_axis(in_delays, sorter, axis=1)
    in_weights = np.take_along_axis(in_weights, sorter, axis=1)
    return in_sources, in_delays, in_weights


def find_last_f_times(f_times, f_sources):
    unique_sources = np.unique(f_sources)
    sorter = np.lexsort((f_times, f_sources))
    indices_last = (
        np.searchsorted(f_sources, unique_sources, side="right", sorter=sorter) - 1
    )
    return f_times[sorter][indices_last], f_sources[sorter][indices_last]


@njit(parallel=True)
def reset_states(f_times, f_sources, st_start, st_c0, st_c1, st_ic0, st_ic1):
    """Reset the states of the neurons that fire.


    Args:
        f_times (npt.NDArray[np.float64]): The firing times for the neurons.
        f_sources (npt.NDArray[np.int64]): The source neurons for the firings.
    """

    for n in prange(f_sources.size):
        l, ft = f_sources[n], f_times[n]
        ipos = np.searchsorted(st_start[:, l], ft, side="left")
        st_start[1:ipos, l] = np.nan
        st_start[-1, l] = ft
        st_c0[-1, l] = REFRACTORY_RESET
        st_ic0[-1, l] = REFRACTORY_RESET

    return st_start, st_c0, st_c1, st_ic0, st_ic1


@njit(parallel=True)
def clean_states(tmin, st_start, st_c0, st_c1, st_ic0, st_ic1):
    """
    Consolidate all states occurring before `tmin` into the last state before `tmin`, and remove (mark as NaN) earlier states.

    This function is designed for per-neuron state histories that are:
    - **Sorted** along the first axis by their start time (`st_start`).
    - **Framed** by an initial state at `-np.inf` (index 0) and a final state at `np.inf` (right before the trailing NaNs).

    For each neuron:
      1. Find the index of the most recent state before `tmin`.
      2. Accumulate contributions from all previous states into that state's coefficients (`st_c0`, `st_c1`).
      3. Mark all earlier states (rows 1 to that index) as removed by setting their `st_start` entries to `NaN`.

    Parameters
    ----------
    tmin : float
        The cutoff time. Only the last state before this time is preserved.
    st_start : ndarray of shape (n_states, n_neurons)
        Start times for each state (must be sorted along axis 0, with NaNs trailing).
    st_c0 : ndarray of shape (n_states, n_neurons)
        First-order cumulative coefficients (will be updated in place).
    st_c1 : ndarray of shape (n_states, n_neurons)
        Second-order cumulative coefficients (will be updated in place).
    st_ic0 : ndarray of shape (n_states, n_neurons)
        First-order independent coefficients (used for recursive updates).
    st_ic1 : ndarray of shape (n_states, n_neurons)
        Second-order independent coefficients (used for recursive updates).

    Returns
    -------
    (st_start, st_c0, st_c1, st_ic0, st_ic1) : tuple of ndarrays
        The same arrays, modified in place:
        - All states before `tmin` are removed except the last one.
        - Coefficients of the preserved state now include contributions from the removed states.
    """
    n_neurons = st_start.shape[1]
    for l in prange(n_neurons):
        ipos = (
            np.searchsorted(st_start[:, l], tmin, side="right") - 1
        )  # always >= 0 since -np.inf is at the start

        st_c0[ipos, l] = st_ic0[ipos, l]
        st_c1[ipos, l] = st_ic1[ipos, l]

        for j in range(1, ipos):
            dt = st_start[ipos, l] - st_start[j, l]

            st_c0[ipos, l] += st_ic0[j, l] * np.exp(-dt)
            st_c0[ipos, l] += st_ic1[j, l] * dt * np.exp(-dt)
            st_c1[ipos, l] += st_ic1[j, l] * np.exp(-dt)

        st_start[1:ipos, l] = np.nan

    return st_start, st_c0, st_c1, st_ic0, st_ic1



def next_f_times(f_thresh, st_start, st_c0, st_c1, st_ic0, st_ic1, min_delays):
    """
    Calculate the next (causally independent) firings produced in the network, among all neurons.

    Warning: This function requires the states to be sorted by st_start.

    Returns:
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]: A tuple containing:
            - f_times: Array of firing times.
            - f_sources: Array of firing sources (neuron indices).
    """
    n_intervals, n_neurons = st_start.shape
    f_times = np.full(n_neurons, np.nan, dtype=np.float64)
    max_f_times = np.full(n_neurons, np.inf, dtype=np.float64)
    n_indices = np.arange(n_neurons)
    nf_indices = n_indices[:]

    for n in range(1, n_intervals - 1):  # st_start
        # early exit if no neuron can possibly fire
        if nf_indices.size == 0:
            break

        # update the coefficients
        prev_length = st_start[n, nf_indices] - st_start[n - 1, nf_indices]
        st_c0[n, nf_indices] = (
            st_c0[n - 1, nf_indices]
            + np.nan_to_num(prev_length) * st_c1[n - 1, nf_indices]
        ) * np.exp(-(prev_length)) + st_ic0[n, nf_indices]
        st_c1[n, nf_indices] = (
            st_c1[n - 1, nf_indices] * np.exp(-(prev_length)) + st_ic1[n, nf_indices]
        )

        # update the candidate firing times
        f_times[nf_indices] = first_crossing(
            st_start[n, nf_indices],
            np.clip(
                st_start[n + 1, nf_indices],
                None,
                max_f_times[nf_indices],
            )
            - st_start[n, nf_indices],
            st_c0[n, nf_indices],
            st_c1[n, nf_indices],
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
        nf_indices = n_indices[np.isnan(f_times) & (st_start[n] < max_f_times)]

    f_sources = np.argwhere(f_times <= max_f_times).flatten()
    f_times = f_times[f_sources]
    return f_times, f_sources


def fire_spikes(
    new_f_times,
    new_f_sources,
    f_thresh,
    f_times,
    f_sources,
    st_start,
    st_c0,
    st_c1,
    st_ic0,
    st_ic1,
    f_thresh_std=0.0,
):
    # Concatenate the new firing times and sources with the existing ones
    f_times = np.concatenate((f_times, new_f_times))
    f_sources = np.concatenate((f_sources, new_f_sources))

    # Flush the input spikes and reset to the refractory state
    st_start, st_c0, st_c1, st_ic0, st_ic1 = reset_states(
        new_f_times, new_f_sources, st_start, st_c0, st_c1, st_ic0, st_ic1
    )

    # Apply threshold noise
    f_thresh[f_sources] = rng.normal(
        FIRING_THRESHOLD, f_thresh_std, size=f_sources.size
    )

    return f_times, f_sources, st_start, st_c0, st_c1, st_ic0, st_ic1


@njit(parallel=True)
def propagate_spikes(
    new_f_times,
    new_f_sources,
    in_sources,
    in_delays,
    in_weights,
    st_start,
    st_c0,
    st_c1,
    st_ic0,
    st_ic1,
):
    """
    Propagate new spikes through the network.
    Add new states at the end of the state arrays...

    Warning: This function assumes that the synaptic inputs are sorted by source neuron. Moreover, the states should be large enough to not overwrite existing states...

    Args:
        new_f_times (_type_): _description_
        new_f_sources (_type_): _description_
        in_sources (_type_): _description_
        in_delays (_type_): _description_
        in_weights (_type_): _description_
        st_start (_type_): _description_
        st_c0 (_type_): _description_
        st_c1 (_type_): _description_
        st_ic0 (_type_): _description_
        st_ic1 (_type_): _description_

    Returns:
        _type_: _description_
    """

    for l in range(in_sources.shape[0]):
        start_idx = np.searchsorted(in_sources[l], new_f_sources, side="left")
        end_idx = np.searchsorted(in_sources[l], new_f_sources, side="right")

        pos = -1
        for n in range(new_f_sources.shape[0]):
            num = end_idx[n] - start_idx[n]
            st_start[pos - num : pos, l] = (
                new_f_times[n] + in_delays[l, start_idx[n] : end_idx[n]]
            )
            # st_c0[pos - num : pos, l] = 0.0  # useless
            # st_c1[pos - num : pos, l] = 0.0  # useless
            st_ic0[pos - num : pos, l] = 0.0
            st_ic1[pos - num : pos, l] = in_weights[l, start_idx[n] : end_idx[n]]
            pos -= num

    return st_start, st_c0, st_c1, st_ic0, st_ic1
