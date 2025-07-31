from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from rsnn.constants import *
from rsnn.utils import setup_logging

logger = setup_logging(__name__, console_level="INFO", file_level="INFO")


def modulo_with_offset(
    x: npt.NDArray[np.float64],
    period: float | np.float64 | npt.NDArray[np.float64],
    offset: float | np.float64 | npt.NDArray[np.float64] = 0.0,
) -> npt.NDArray[np.float64]:
    """
    Apply modulo operation with offset, so that the result is in the range [offset, offset + period).

    Args:
        x (npt.NDArray[np.float64]): Input array.
        period (float | np.float64 | npt.NDArray[np.float64]): Period for the modulo operation.
        offset (float | np.float64 | npt.NDArray[np.float64], optional): Offset to apply. Defaults to 0.0.

    Returns:
        npt.NDArray[np.float64]: Resulting array after applying modulo with offset.
    """

    return x - period * np.floor_divide(x - offset, period)


def compute_states(
    in_times: npt.NDArray[np.float64],
    in_channels: npt.NDArray[np.int64],
    c0_in_tmin: npt.NDArray[np.float64],
    c1_in_tmin: npt.NDArray[np.float64],
    tmin: float | np.float64 = -np.inf,
    tmax: float | np.float64 = np.inf,
    times: Optional[npt.NDArray[np.float64]] = None,
) -> Tuple[
    npt.NDArray[np.float64],  # start: shape (n_intervals)
    npt.NDArray[np.float64],  # c0_in: shape (n_intervals, n_in_channels + 1)
    npt.NDArray[np.float64],  # c1_in: shape (n_intervals, n_in_channels + 1)
]:
    """
    Compute the coefficients (c0nk and c1nk) defining the states of every input (indexed by k) for any time between tmin and tmax, on disjoint intervals (indexed by n).
    The intervals partition the time range [tmin, tmax] in n_intervals = in_times.size + 1 intervals from the following time markers:
    - tmin, the start of the time range
    - in_times, the input spike times in [tmin, tmax]
    The intervals are reconstructed from their start and length.
    The signal (c0nk + c1nk * dt) * exp(-dt) for 0 <= dt < length[n] then corresponds to
    a) the kth input signal on the nth interval [start[n], start[n] + length[n]) if start[n] + length[n] < bf_time.
    b) the derivative of the kth input signal on the nth interval [start[n], start[n] + length[n]) if start[n] + length[n] < f_time and start[n] >= f_time.
    c) the kth input signal on the nth interval [start[n], start[n] + length[n]) if start[n] = f_time.

    Args:
        f_time (np.double): _description_
        bf_time (np.double): _description_
        in_times (np.ndarray): _description_
        in_channels (np.ndarray): _description_
        n_in_channels (np.intp): _description_
        zmax (np.double): _description_
        dzmin (np.double): _description_

    Returns:
        Tuple[npt.NDArray[np.float64], ...]: A tuple containing:
            - start: shape (n_intervals)
            - length: shape (n_intervals)
            - c0_in: shape (n_intervals, n_in_channels + 1)
            - c1_in: shape (n_intervals, n_in_channels + 1)
    """
    # Extract the input spikes that contribute to the neuron state in the interval [tmin, tmax]
    select = (in_times >= tmin) & (in_times <= tmax)
    in_times = in_times[select]
    in_channels = in_channels[select]

    # Extract the times in the range [tmin, tmax] if provided
    times = (
        times[(times >= tmin) & (times <= tmax)] if times is not None else np.array([])
    )

    # Initialize the start arrays
    start = np.concatenate((np.array([tmin]), in_times, times))

    # Initialize the coefficients arrays
    c0_in = np.zeros((start.size, c0_in_tmin.size))
    c1_in = np.zeros((start.size, c1_in_tmin.size))
    c0_in[0] = c0_in_tmin
    c1_in[0] = c1_in_tmin
    c1_in[np.arange(1, in_times.size + 1), in_channels] = 1.0

    # Sort the start array
    sorter = np.argsort(start)
    start = start[sorter]
    c0_in = c0_in[sorter]
    c1_in = c1_in[sorter]

    # Input signals for the potential
    for n in range(start.size - 1):
        c0_in[n + 1] += (c0_in[n] + c1_in[n] * (start[n + 1] - start[n])) * np.exp(
            -(start[n + 1] - start[n])
        )
        c1_in[n + 1] += c1_in[n] * np.exp(-(start[n + 1] - start[n]))

    return start, c0_in, c1_in


def find_maximum_violation(
    c0: npt.NDArray[np.float64],
    c1: npt.NDArray[np.float64],
    length: npt.NDArray[np.float64],
    lim: npt.NDArray[np.float64],
) -> Optional[Tuple[np.float64, np.intp, np.float64]]:
    """
    Compute the maximum violation of the condition c0[n] + c1[n] * dt * exp(-dt) > lim[n] for 0 <= dt < length[n] for each interval n, if any.

    Args:
        c0 (npt.NDArray[np.float64]): the 0th coefficients for each interval.
        c1 (npt.NDArray[np.float64]): the 1st coefficients for each interval.
        length (npt.NDArray[np.float64]): the length for each interval.
        lim (npt.NDArray[np.float64]): the maximum allowed value for each interval.

    Returns:
        Tuple[np.float64, np.intp, np.float64]: the maximum violation and the interval index nmax and the time difference dt in [0, length_nmax] at which the violation occurs.
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
        # logger.debug(
        #     f"Maximum violation (with value {dv[imax]}) found for the {imax[0]}th interval at dt = {dt[imax]}."
        # )
        return dv[imax], imax[1], dt[imax]

    # logger.debug("No violation found.")
    return None
