# import numba as nb
import numpy as np
from numba import jit, njit, prange
from numba.typed import List

from rsnn.constants import FIRING_THRESHOLD, REFRACTORY_RESET
from rsnn.sim.utils import first_crossing, floyd_warshall
from rsnn.utils import setup_logging

logger = setup_logging(__name__, console_level="INFO", file_level="INFO")


@njit(parallel=True)
def clean_states(n_neurons, st_start, st_c0, st_c1, st_dc0, st_dc1, tmin):
    """
    Clean the states by removing those that end before the given time, while ensuring validity of the first state.
    Note 1: the states necessarily contain at least two states: the initial state and the final state.

    Args:
        n_neurons (int): The number of neurons.
        st_start (list): List of start times for each neuron.
        st_c0 (list): List of c0 values for each neuron.
        st_c1 (list): List of c1 values for each neuron.
        st_dc0 (list): List of dc0 values for each neuron.
        st_dc1 (list): List of dc1 values for each neuron.
        tmin (float): The time before which states should be removed.
    """
    for n in prange(n_neurons):
        ipos = np.searchsorted(st_start[n], tmin, side="right") - 1  # always >= 0

        if ipos > 0:  # if there are states to clean
            for i in range(1, ipos + 1):
                dt = np.nan_to_num(st_start[n][i] - st_start[n][i - 1])
                st_c0[i] = (st_c0[i - 1] + dt * st_c1[i - 1]) * np.exp(-dt) + st_dc0[i]
                st_c1[i] = st_c1[i - 1] * np.exp(-dt) + st_dc1[i]

            st_start[n] = st_start[n][ipos - 1 :]
            st_c0 = st_c0[ipos - 1 :]
            st_c1 = st_c1[ipos - 1 :]
            st_dc0 = st_dc0[ipos - 1 :]
            st_dc1 = st_dc1[ipos - 1 :]

            st_start[0] = -np.inf
            st_c0[0] = 0.0
            st_c1[0] = 0.0
            st_dc0[0] = 0.0
            st_dc1[0] = 0.0

            st_dc0[1] = st_c0[1]
            st_dc1[1] = st_c1[1]


@njit(parallel=True)
def compute_next_f_times(
    n_neurons,
    f_threshold,
    st_start,
    st_c0,
    st_c1,
    st_dc0,
    st_dc1,
    in_min_delays,
):
    # tmax = np.inf
    # logger.debug(f"New step started.")
    f_times = np.full(n_neurons, np.nan, dtype=np.float64)

    for n_id in prange(n_neurons):
        logger.debug(f"Processing neuron {n_id}.")
        tmax = np.nanmin(f_times + in_min_delays[n_id], initial=np.inf)

        for st_id in range(1, st_start[n_id].shape[0] - 1):
            if st_start[n_id][st_id] < tmax:
                length = st_start[n_id][st_id] - st_start[n_id][st_id - 1]
                st_c0[n_id][st_id] = (
                    st_c0[n_id][st_id - 1]
                    + np.nan_to_num(length) * st_c1[n_id][st_id - 1]
                ) * np.exp(-length) + st_dc0[n_id][st_id]
                st_c1[n_id][st_id] = (
                    st_c1[n_id][st_id - 1] * np.exp(-length) + st_dc1[n_id][st_id]
                )

                t = first_crossing(
                    st_start[n_id][st_id],
                    st_start[n_id][st_id + 1] - st_start[n_id][st_id],
                    st_c0[n_id],
                    st_c1[n_id],
                    f_threshold[n_id],
                )
                # t = first_crossing(self.states[n], self.threshold)
                if np.isfinite(t):
                    f_times[n_id] = (
                        t if t < tmax else np.nan
                    )  # replace with masked operation
                    break
            else:
                break

        # f_time = neuron.step(tmin, tmax) # tmax is based on the delay
        # if f_time is not None:
        #     (src_id, tmax) = (id, f_time)

    # Accept the largest possible number of independent spikes
    f_sources = np.argwhere(
        np.all(
            f_times[:, None]
            <= np.nan_to_num(f_times[None, :], nan=np.inf) + in_min_delays,
            axis=1,
        )
    ).flatten()
    f_times = f_times[f_sources]
    logger.debug(f"New spikes at {f_times} from {f_sources}")
    return f_times, f_sources
    # self.propagate_spikes(f_times, f_sources, std_threshold)

    # return np.min(f_times, initial=np.inf)


@njit(parallel=True)
def fire_spikes(
    f_times,
    new_f_times,
    new_f_sources,
    st_start,
    st_c0,
    st_c1,
    st_dc0,
    st_dc1,
    f_thresh,
    f_thresh_noise,
):
    for i in prange(new_f_sources.size):
        n_id = new_f_sources[i]
        new_f_time = new_f_times[n_id]

        # Add the firing time to the list of firing times
        f_times[n_id] = np.sort(np.append(f_times[n_id], new_f_time))

        # Add threshold noise
        f_thresh[n_id] = FIRING_THRESHOLD + f_thresh_noise[n_id]

        # Enter refractory period
        ipos = np.searchsorted(st_start[n_id], new_f_time, side="left")  # always >= 1
        if ipos > 1:
            st_start[n_id] = st_start[n_id][(ipos - 2) :]
            st_c0[n_id] = st_c0[n_id][(ipos - 2) :]
            st_c1[n_id] = st_c1[n_id][(ipos - 2) :]
            st_dc0[n_id] = st_dc0[n_id][(ipos - 2) :]
            st_dc1[n_id] = st_dc1[n_id][(ipos - 2) :]

            st_start[n_id][0] = -np.inf
            st_c0[n_id][0] = 0.0
            st_c1[n_id][0] = 0.0
            st_dc0[n_id][0] = 0.0
            st_dc1[n_id][0] = 0.0
        else:
            st_start[n_id] = np.concatenate((np.array([-np.inf]), st_start[n_id]))
            st_c0[n_id] = np.concatenate((np.array([0.0]), st_c0[n_id]))
            st_c1[n_id] = np.concatenate((np.array([0.0]), st_c1[n_id]))
            st_dc0[n_id] = np.concatenate((np.array([0.0]), st_dc0[n_id]))
            st_dc1[n_id] = np.concatenate((np.array([0.0]), st_dc1[n_id]))

        st_start[n_id][1] = new_f_time
        st_c0[n_id][1] = REFRACTORY_RESET
        st_c1[n_id][1] = 0.0
        st_dc0[n_id][1] = REFRACTORY_RESET
        st_dc1[n_id][1] = 0.0


@njit(parallel=True)
def propagate_spikes(
    n_neurons,
    new_f_times,
    new_f_sources,
    in_sources,
    in_delays,
    in_weights,
    st_start,
    st_c0,
    st_c1,
    st_dc0,
    st_dc1,
):
    for n_id in prange(n_neurons):
        indices = np.argwhere(new_f_sources[:, None] == in_sources[n_id][None, :])
        new_st_start = new_f_times[indices[:, 0]] + in_delays[0, indices[:, 1]]
        new_st_dc1 = in_weights[n_id, indices[:, 1]]

        st_start[n_id] = np.concatenate((st_start[n_id], new_st_start), axis=0)
        st_c0[n_id] = np.concatenate((st_c0[n_id], np.zeros_like(new_st_start)), axis=0)
        st_c1[n_id] = np.concatenate((st_c1[n_id], np.zeros_like(new_st_start)), axis=0)
        st_dc0[n_id] = np.concatenate(
            (st_dc0[n_id], np.zeros_like(new_st_start)), axis=0
        )
        st_dc1[n_id] = np.concatenate((st_dc1[n_id], new_st_dc1), axis=0)

        sorter = np.argsort(st_start[n_id])
        st_start[n_id] = st_start[n_id][sorter]
        st_c0[n_id] = st_c0[n_id][sorter]
        st_c1[n_id] = st_c1[n_id][sorter]
        st_dc0[n_id] = st_dc0[n_id][sorter]
        st_dc1[n_id] = st_dc1[n_id][sorter]


@jit(parallel=True)
def init_states(n_neurons: int, f_times, in_sources, in_delays, in_weights):
    st_start = List(
        [np.array([-np.inf, np.inf], dtype=np.float64) for _ in range(n_neurons)]
    )
    st_c0 = List([np.zeros((2,), dtype=np.float64) for _ in range(n_neurons)])
    st_c1 = List([np.zeros((2,), dtype=np.float64) for _ in range(n_neurons)])
    st_dc0 = List([np.zeros((2,), dtype=np.float64) for _ in range(n_neurons)])
    st_dc1 = List([np.zeros((2,), dtype=np.float64) for _ in range(n_neurons)])

    n_f_times = sum([ft.size for ft in f_times])
    flat_f_times = np.empty(n_f_times, dtype=np.float64)
    flat_f_sources = np.empty(n_f_times, dtype=np.int64)
    start_idx = 0
    for n_id in range(n_neurons):
        if f_times[n_id].size > 0:
            end_idx = start_idx + f_times[n_id].size
            flat_f_times[start_idx:end_idx] = f_times[n_id]
            flat_f_sources[start_idx:end_idx] = n_id
            start_idx = end_idx
    print(f"Flat firing times: {flat_f_times}, sources: {flat_f_sources}")

    for n_id in prange(n_neurons):
        indices = np.argwhere(flat_f_sources[:, None] == in_sources[n_id][None, :])
        new_st_start = flat_f_times[indices[:, 0]] + in_delays[0, indices[:, 1]]
        new_st_dc1 = in_weights[n_id, indices[:, 1]]

        st_start[n_id] = np.concatenate((st_start[n_id], new_st_start), axis=0)
        st_c0[n_id] = np.concatenate((st_c0[n_id], np.zeros_like(new_st_start)), axis=0)
        st_c1[n_id] = np.concatenate((st_c1[n_id], np.zeros_like(new_st_start)), axis=0)
        st_dc0[n_id] = np.concatenate(
            (st_dc0[n_id], np.zeros_like(new_st_start)), axis=0
        )
        st_dc1[n_id] = np.concatenate((st_dc1[n_id], new_st_dc1), axis=0)

        sorter = np.argsort(st_start[n_id])
        st_start[n_id] = st_start[n_id][sorter]
        st_c0[n_id] = st_c0[n_id][sorter]
        st_c1[n_id] = st_c1[n_id][sorter]
        st_dc0[n_id] = st_dc0[n_id][sorter]
        st_dc1[n_id] = st_dc1[n_id][sorter]

        if f_times[n_id].size > 0:
            # Ensure the last firing time is included in the states
            last_f_time = f_times[n_id][-1]
            ipos = np.searchsorted(
                st_start[n_id], last_f_time, side="left"
            )  # always >= 1
            if ipos > 1:
                st_start[n_id] = st_start[n_id][(ipos - 2) :]
                st_c0[n_id] = st_c0[n_id][(ipos - 2) :]
                st_c1[n_id] = st_c1[n_id][(ipos - 2) :]
                st_dc0[n_id] = st_dc0[n_id][(ipos - 2) :]
                st_dc1[n_id] = st_dc1[n_id][(ipos - 2) :]

                st_start[n_id][0] = -np.inf
                st_c0[n_id][0] = 0.0
                st_c1[n_id][0] = 0.0
                st_dc0[n_id][0] = 0.0
                st_dc1[n_id][0] = 0.0
            else:
                st_start[n_id] = np.concatenate((np.array([-np.inf]), st_start[n_id]))
                st_c0[n_id] = np.concatenate((np.array([0.0]), st_c0[n_id]))
                st_c1[n_id] = np.concatenate((np.array([0.0]), st_c1[n_id]))
                st_dc0[n_id] = np.concatenate((np.array([0.0]), st_dc0[n_id]))
                st_dc1[n_id] = np.concatenate((np.array([0.0]), st_dc1[n_id]))

            st_start[n_id][1] = last_f_time
            st_c0[n_id][1] = REFRACTORY_RESET
            st_c1[n_id][1] = 0.0
            st_dc0[n_id][1] = REFRACTORY_RESET
            st_dc1[n_id][1] = 0.0

    return st_start, st_c0, st_c1, st_dc0, st_dc1


def sim(
    n_neurons,
    f_times,
    f_thresh,
    in_sources,
    in_delays,
    in_weights,
    st_start,
    st_c0,
    st_c1,
    st_dc0,
    st_dc1,
    tmin,
    tmax,
    f_thresh_std,
    in_min_delays=None,
    rng=None,
):
    if in_min_delays is None:
        in_min_delays = floyd_warshall(in_sources, in_delays)
    if rng is None:
        rng = np.random.default_rng()

    time = tmin
    while time < tmax:
        clean_states(n_neurons, st_start, st_c0, st_c1, st_dc0, st_dc1, time)
        new_f_times, new_f_sources = compute_next_f_times(
            n_neurons, f_thresh, st_start, st_c0, st_c1, st_dc0, st_dc1, in_min_delays
        )
        fire_spikes(
            new_f_times,
            new_f_sources,
            f_times,
            st_start,
            st_c0,
            st_c1,
            st_dc0,
            st_dc1,
            f_thresh,
            rng.normal(0, f_thresh_std, n_neurons),  # threshold noise
        )
        propagate_spikes(
            n_neurons,
            new_f_times,
            new_f_sources,
            in_sources,
            in_delays,
            in_weights,
            st_start,
            st_c0,
            st_c1,
            st_dc0,
            st_dc1,
        )
        time = np.min(new_f_times)
