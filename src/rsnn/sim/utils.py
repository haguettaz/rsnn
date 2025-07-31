from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.special import lambertw


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
    start : ndarray of shape (n_states,)
        Start times for each state.
    length : ndarray of shape (n_states,)
        Lengths of the states for each state.
    c0 : ndarray of shape (n_states,)
        First-order coefficients for each state.
    c1 : ndarray of shape (n_states,)
        Second-order coefficients for each state.
    f_thresh : ndarray of shape (n_states,)
        Firing threshold for each state.

    Returns
    -------
    np.ndarray
        First crossing times for each state.
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


# def first_crossing(
#     start: float,
#     length: float,
#     c0: float,
#     c1: float,
#     f_thresh: float,
# ) -> np.float64 | float:

#     if c0 < f_thresh:
#         if c1 > 0:
#             dt = -(lambertw(-f_thresh / c1 * np.exp(-c0 / c1), 0)) - c0 / c1
#             if np.isreal(dt) and (dt >= 0.0) and (dt < length):
#                 return start + dt.real
#         elif c1 < 0:
#             dt = (
#                 -(
#                     lambertw(
#                         -f_thresh / c1 * np.exp(-c0 / c1),
#                         -1,
#                     )
#                 )
#                 - c0 / c1
#             )
#             if np.isreal(dt) and (dt >= 0.0) and (dt < length):
#                 return np.float64(start + dt.real)
#         elif f_thresh < 0:
#             dt = np.log(c0 / f_thresh)
#             if (dt >= 0.0) and (dt < length):
#                 return start + dt
#         return np.nan

#     else:
#         return start


# def first_crossing_vectorized(
#     start: npt.NDArray[np.float64],
#     length: npt.NDArray[np.float64],
#     c0: npt.NDArray[np.float64],
#     c1: npt.NDArray[np.float64],
#     f_thresh: npt.NDArray[np.float64],
# ) -> np.ndarray:
#     """Vectorized version of first_crossing."""

#     dt0 = -(lambertw(-f_thresh / c1 * np.exp(-c0 / c1), 0)) - c0 / c1
#     dt1 = -(lambertw(-f_thresh / c1 * np.exp(-c0 / c1), -1)) - c0 / c1
#     dt2 = np.log(c0 / f_thresh)

#     next_f_times = np.where(
#         c0 < f_thresh,
#         np.where(
#             c1 > 0,
#             np.where(
#                 np.isreal(dt0) & (dt0 >= 0.0) & (dt0 < length), start + dt0, np.nan
#             ),
#             np.where(
#                 c1 < 0,
#                 np.where(
#                     np.isreal(dt1) & (dt1 >= 0.0) & (dt1 < length), start + dt1, np.nan
#                 ),
#                 np.where((dt2 >= 0.0) & (dt2 < length), start + dt2, np.nan),
#             ),
#         ),
#         start,
#     )
#     return next_f_times


# def floyd_warshall(
#     in_sources: np.ndarray,
#     in_delays: np.ndarray,
# ) -> np.ndarray:
#     """Compute the minimum delays between all pairs of neurons using the Floyd-Warshall algorithm.
#     Note: The returned array of delays is indexed by (target, source).

#     Args:
#         in_sources (np.ndarray): Array of source neuron IDs for each connection, with shape (L, K).
#         in_delays (np.ndarray): Array of delays for each connection, with shape (L, K).

#     Returns:
#         np.ndarray: Array of minimum delays between all pairs of neurons, with shape (L, L).
#     """
#     n_nodes, _ = in_sources.shape

#     indices = np.arange(n_nodes)
#     select = in_sources[:, :, None] == indices[None, None, :]
#     min_in_delays = np.min(
#         np.broadcast_to(in_delays[:, :, None], select.shape),
#         where=select,
#         initial=np.inf,
#         axis=1,
#     )

#     for k in range(n_nodes):
#         min_in_delays = np.minimum(
#             min_in_delays,
#             min_in_delays[:, k][:, None] + min_in_delays[k, :],
#         )

#     return min_in_delays
