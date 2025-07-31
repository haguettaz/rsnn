from typing import Tuple

import numpy as np
import numpy.typing as npt


def sort_states(
    st_start: npt.NDArray[np.float64],
    st_c0: npt.NDArray[np.float64],
    st_c1: npt.NDArray[np.float64],
    st_ic0: npt.NDArray[np.float64],
    st_ic1: npt.NDArray[np.float64],
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """
    Sort all states of each neuron by their start times (`st_start`).
    The order of the other state arrays (`st_c0`, `st_c1`, `st_ic0`, `st_ic1`) is adjusted accordingly.

    The sort order is:
    - `-np.inf` (initial state)
    - All finite values (sorted ascending)
    - `np.inf` (final state)
    - `NaN` (removed states)

    Parameters
    ----------
    st_start : ndarray of shape (n_states, n_neurons)
        Start times for each state (must be sorted along axis 0).
    st_c0 : ndarray of shape (n_states, n_neurons)
        First-order cumulative coefficients.
    st_c1 : ndarray of shape (n_states, n_neurons)
        Second-order cumulative coefficients.
    st_ic0 : ndarray of shape (n_states, n_neurons)
        First-order independent coefficients.
    st_ic1 : ndarray of shape (n_states, n_neurons)
        Second-order independent coefficients.

    Returns
    -------
    (st_start, st_c0, st_c1, st_ic0, st_ic1) : tuple of ndarrays
        A sorted copy of the provided arrays.
    """
    sorter = np.argsort(st_start, axis=0)
    st_start = np.take_along_axis(st_start, sorter, axis=0)
    st_c0 = np.take_along_axis(st_c0, sorter, axis=0)
    st_c1 = np.take_along_axis(st_c1, sorter, axis=0)
    st_ic0 = np.take_along_axis(st_ic0, sorter, axis=0)
    st_ic1 = np.take_along_axis(st_ic1, sorter, axis=0)
    return st_start, st_c0, st_c1, st_ic0, st_ic1
