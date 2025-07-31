import json
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from numba import njit, prange
from scipy.sparse.csgraph import floyd_warshall
from scipy.special import lambertw

from rsnn.constants import FIRING_THRESHOLD, REFRACTORY_RESET
from rsnn.sim.neuron import Neuron
from rsnn.sim.utils import first_crossing_vectorized
from rsnn.utils import setup_logging

logger = setup_logging(__name__, console_level="INFO", file_level="INFO")

BUFFER_SIZE: int = 1000  # Size of the buffer for neuron states


@njit(parallel=True)
def clean_states(tmin, st_start, st_c0, st_c1, st_dc0, st_dc1):
    n_neurons = st_start.shape[1]

    for l in prange(n_neurons):
        ipos = (
            np.searchsorted(st_start[:, l], tmin, side="right") - 1
        )  # always >= 0 since -np.inf is at the start

        st_c0[ipos, l] = st_dc0[ipos, l]
        st_c1[ipos, l] = st_dc1[ipos, l]

        for j in range(1, ipos):
            dt = st_start[ipos, l] - st_start[j, l]

            st_c0[ipos, l] += st_dc0[j, l] * np.exp(-dt)
            st_c0[ipos, l] += st_dc1[j, l] * dt * np.exp(-dt)
            st_c1[ipos, l] += st_dc1[j, l] * np.exp(-dt)

        st_start[1:ipos, l] = np.nan

    return st_start, st_c0, st_c1, st_dc0, st_dc1


def sort_states(st_start, st_c0, st_c1, st_dc0, st_dc1):
    sorter = np.argsort(st_start, axis=0)
    st_start = np.take_along_axis(st_start, sorter, axis=0)
    st_c0 = np.take_along_axis(st_c0, sorter, axis=0)
    st_c1 = np.take_along_axis(st_c1, sorter, axis=0)
    st_dc0 = np.take_along_axis(st_dc0, sorter, axis=0)
    st_dc1 = np.take_along_axis(st_dc1, sorter, axis=0)
    return st_start, st_c0, st_c1, st_dc0, st_dc1


class Simulator:
    """Simulator for a network of neurons."""

    def __init__(
        self,
        n_neurons: int,
        in_sources: npt.NDArray[np.int64],
        in_delays: npt.NDArray[np.float64],
        in_weights: npt.NDArray[np.float64],
    ):
        """
        Initialize the simulator with a list of neurons and connections between them.
        Each neuron has the same number of incoming connections.

        Args:
            n_neurons (int): Number of neurons in the network.
            in_sources (_type_): Array of source neuron IDs for each connection, with shape (L, K).
            in_delays (_type_): Array of delays for each connection, with shape (L, K).
            in_weights (_type_): Array of weights for each connection, with shape (L, K).
        """
        self.n_neurons = n_neurons
        self.n_indices = np.arange(n_neurons)

        # Initialize neuron data
        self.last_f_times: npt.NDArray[np.float64] = np.full(
            n_neurons, -np.inf, dtype=np.float64
        )
        self.f_thresh: npt.NDArray[np.float64] = np.full(
            n_neurons, FIRING_THRESHOLD, dtype=np.float64
        )

        # Neuron states buffer
        self.st_start = np.full((BUFFER_SIZE, n_neurons), np.nan, dtype=np.float64)
        self.st_start[0, :] = -np.inf  # Initial state
        self.st_start[1, :] = np.inf  # Final state
        self.st_sorter: npt.NDArray[np.int64] = np.argsort(self.st_start, axis=0)
        self.st_c0 = np.zeros((BUFFER_SIZE, n_neurons), dtype=np.float64)  # c0 values
        self.st_c1 = np.zeros((BUFFER_SIZE, n_neurons), dtype=np.float64)  # c1 values
        self.st_dc1 = np.zeros((BUFFER_SIZE, n_neurons), dtype=np.float64)  # dc1 values
        self.st_dc0 = np.zeros((BUFFER_SIZE, n_neurons), dtype=np.float64)  # dc0 values

        # Initialize firing times and sources of these firings
        self.f_times: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self.f_sources: npt.NDArray[np.int64] = np.array([], dtype=np.int64)

        # Initialize connections between neurons
        self.in_sources = in_sources
        self.in_delays = in_delays
        self.in_weights = in_weights

        # Initialize the minimum delay between every pair of neurons
        # This is used to accept as many causally independent spikes as possible
        indices = np.arange(n_neurons)
        select = in_sources[:, :, None] == indices[None, None, :]
        self.in_min_delays = np.min(
            np.broadcast_to(in_delays[:, :, None], select.shape),
            where=select,
            initial=np.inf,
            axis=1,
        )
        floyd_warshall(self.in_min_delays, directed=True, overwrite=True)

    @property
    def n_connections(self) -> int:
        """Get the number of connections in the network.

        Returns:
            int: the number of connections in the network.
        """
        return self.in_sources.size

    # def clean_states(self, tmin: float):
    #     """Remove all states that are before tmin and update the coefficients.

    #     Args:
    #         tmin (float): The minimum time to keep in the states.
    #     """
    #     # search sorted of tmin in st_start
    #     ipos = np.apply_along_axis(
    #         np.searchsorted, 0, self.st_start, tmin, side="right"
    #     )

    # def clear_states(
    #     self, f_times: npt.NDArray[np.float64], f_sources: npt.NDArray[np.int64]
    # ):
    #     """Clear the states of the neurons that fire.

    #     Args:
    #         f_times (npt.NDArray[np.float64]): The firing times for the neurons.
    #         f_sources (npt.NDArray[np.int64]): The source neurons for the firings.
    #     """
    #     mask = (self.st_start[:, f_sources] < f_times[None, :]) & np.isfinite(
    #         self.st_start[:, f_sources]
    #     )
    #     self.st_start[:, f_sources] = np.where(
    #         mask, np.nan, self.st_start[:, f_sources]
    #     )
    #     self.st_c0[:, f_sources] = np.where(mask, 0.0, self.st_c0[:, f_sources])
    #     self.st_c1[:, f_sources] = np.where(mask, 0.0, self.st_c1[:, f_sources])
    #     self.st_dc0[:, f_sources] = np.where(mask, 0.0, self.st_dc0[:, f_sources])
    #     self.st_dc1[:, f_sources] = np.where(mask, 0.0, self.st_dc1[:, f_sources])

    #     # Refractory mechanism -> needs to make sure we do not overwrite something, nan are stored at the end
    #     self.st_start[-1, f_sources] = f_times
    #     self.st_c0[-1, f_sources] = REFRACTORY_RESET
    #     self.st_dc0[-1, f_sources] = REFRACTORY_RESET

    #     self.st_sorter = np.argsort(self.st_start, axis=0)
    #     self.st_start = np.take_along_axis(self.st_start, self.st_sorter, axis=0)
    #     self.st_c0 = np.take_along_axis(self.st_c0, self.st_sorter, axis=0)
    #     self.st_c1 = np.take_along_axis(self.st_c1, self.st_sorter, axis=0)
    #     self.st_dc0 = np.take_along_axis(self.st_dc0, self.st_sorter, axis=0)
    #     self.st_dc1 = np.take_along_axis(self.st_dc1, self.st_sorter, axis=0)

    def next_f_times(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """Calculate the next firings produced in the network, among all neurons.

        Returns:
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]: A tuple containing:
                - f_times: Array of firing times.
                - f_sources: Array of firing sources (neuron indices).
        """
        f_times = np.full(self.n_neurons, np.nan, dtype=np.float64)
        max_f_times = np.full(self.n_neurons, np.inf, dtype=np.float64)
        nf_indices = self.n_indices[:]

        for n in range(1, self.st_start.shape[0] - 1):  # first state starts at -inf
            # early exit if no neuron can possibly fire
            if nf_indices.size == 0:
                break

            state_idx = self.st_sorter[n][nf_indices]
            prev_state_idx = self.st_sorter[n - 1][nf_indices]
            next_state_idx = self.st_sorter[n + 1][nf_indices]

            # update the coefficients
            self.st_c0[state_idx, nf_indices] = (
                self.st_c0[prev_state_idx, nf_indices]
                + np.nan_to_num(
                    self.st_start[state_idx, nf_indices]
                    - self.st_start[prev_state_idx, nf_indices]
                )
                * self.st_c1[prev_state_idx, nf_indices]
            ) * np.exp(
                -(
                    self.st_start[state_idx, nf_indices]
                    - self.st_start[prev_state_idx, nf_indices]
                )
            ) + self.st_dc0[
                state_idx, nf_indices
            ]

            self.st_c1[state_idx, nf_indices] = (
                self.st_c1[prev_state_idx, nf_indices]
                * np.exp(
                    -(
                        self.st_start[state_idx, nf_indices]
                        - self.st_start[prev_state_idx, nf_indices]
                    )
                )
                + self.st_dc1[state_idx, nf_indices]
            )

            # update the candidate firing times
            f_times[nf_indices] = first_crossing_vectorized(
                self.st_start[state_idx, nf_indices],
                np.clip(
                    self.st_start[next_state_idx, nf_indices],
                    None,
                    max_f_times[nf_indices],
                )
                - self.st_start[state_idx, nf_indices],
                self.st_c0[state_idx, nf_indices],
                self.st_c1[state_idx, nf_indices],
                self.f_thresh[nf_indices],
            )

            # update the maximum firing times
            max_f_times = np.nanmin(
                np.nan_to_num(f_times[None, :], nan=np.inf)
                + self.in_min_delays,  # can be further optimized...
                initial=np.inf,
                axis=1,
            )

            # update the indices of the neurons that can possibly fire
            nf_indices = self.n_indices[
                np.isnan(f_times) & (self.st_start[self.st_sorter[n]] < max_f_times)
            ]

        f_sources = np.argwhere(f_times <= max_f_times).flatten()
        f_times = f_times[f_sources]
        return f_times, f_sources

    def run(self, start: float, end: float, std_threshold: float = 0.0):
        time = start
        while time < end:
            time = self.step(time, std_threshold)

    def step(self, tmin: float, std_threshold: float = 0.0) -> np.float64:
        # needs to update the states based on tmin

        # remove all states that are before tmin and update the coefficients
        self.clean_states(tmin)
        f_times, f_sources = self.next_f_times()

        logger.debug(f"New spikes at {f_times} from {f_sources}")
        self.propagate_spikes(f_times, f_sources, std_threshold)

        return np.min(f_times, initial=np.inf)

    def propagate_spikes(self, f_times, f_sources, std_threshold: float = 0.0):
        for f_src, f_time in zip(f_sources, f_times):
            # Make the neurons fire
            self.neurons[f_src].fire(f_time, np.random.normal(0, std_threshold))

            # Propagate the spikes to the target neurons, by updating their states
            for l in range(self.n_neurons):
                select_from = self.in_sources[l] == f_src
                starts = f_time + self.in_delays[l][select_from]
                dc1s = self.in_weights[l][select_from]
                self.neurons[l].add_states(
                    starts,
                    np.full_like(starts, np.inf),
                    np.zeros_like(starts),
                    np.zeros_like(starts),
                    np.zeros_like(starts),
                    dc1s,
                )

        # for tgt_id, tgt_neuron in enumerate(self.neurons):
        #     starts = np.array(
        #         [f_time + conn[0] for conn in self.connections[(src_id, tgt_id)]]
        #     )
        #     dc1s = np.array([conn[1] for conn in self.connections[(src_id, tgt_id)]])
        #     tgt_neuron.add_states(
        #         starts,
        #         np.full_like(starts, np.inf),
        #         np.zeros_like(starts),
        #         np.zeros_like(starts),
        #         np.zeros_like(starts),
        #         dc1s,
        #     )

    def init_from_f_times(self):
        """Initialize the neurons' states based on their firing times."""
        for tgt_id, tgt_neuron in enumerate(self.neurons):
            tgt_neuron.clear_states()
            starts = np.concatenate(
                [
                    (
                        self.neurons[src_id].f_times[None, :]
                        + self.in_delays[tgt_id][self.in_sources[tgt_id] == src_id][
                            :, None
                        ]
                    ).reshape(-1)
                    for src_id in range(self.n_neurons)
                ]
            )
            dc1s = np.concatenate(
                [
                    np.repeat(
                        self.in_weights[tgt_id][self.in_sources[tgt_id] == src_id],
                        self.neurons[src_id].f_times.size,
                    )
                    for src_id in range(self.n_neurons)
                ]
            )
            # starts = np.array([
            #     f_time + conn[0]
            #     for (src_id, scr_neuron) in enumerate(self.neurons)
            #     for f_time in scr_neuron.f_times
            #     for conn in self.connections[(src_id, tgt_id)]
            # ])

            # dc1s = np.array([
            #     conn[1]
            #     for (src_id, scr_neuron) in enumerate(self.neurons)
            #     for _ in scr_neuron.f_times
            #     for conn in self.connections[(src_id, tgt_id)]
            # ])

            tgt_neuron.add_states(
                starts,
                np.full_like(starts, np.inf),
                np.zeros_like(starts),
                np.zeros_like(starts),
                np.zeros_like(starts),
                dc1s,
            )

            if tgt_neuron.f_times.size > 0:
                tgt_neuron.recover(tgt_neuron.f_times[-1])

    # def to_dict(self) -> dict:
    #     """Convert the network to a dictionary for JSON serialization"""
    #     return {
    #         "neurons": [
    #             {
    #                 "threshold": neuron.threshold,
    #                 "f_times": neuron.f_times.tolist(),
    #                 "sources": self.in_sources[id].tolist(),
    #                 "delays": self.in_delays[id].tolist(),
    #                 "weights": self.in_weights[id].tolist(),
    #             }
    #             for id, neuron in enumerate(self.neurons)
    #         ],
    #         # "co_sources": self.in_sources.tolist(),
    #         # "co_delays": self.in_delays.tolist(),
    #         # "co_weights": self.in_weights.tolist(),
    #         # "connections": {
    #         #     f"{source_id},{target_id}": [
    #         #         {
    #         #             "delay": conn[0],
    #         #             "weight": conn[1],
    #         #         }
    #         #         for conn in conns
    #         #     ]
    #         #     for (source_id, target_id), conns in self.connections.items()
    #         # },
    #     }

    # @classmethod
    # def from_dict(cls, data: dict) -> "Simulator":
    #     """Create a network from a dictionary loaded from JSON"""
    #     # Restore neurons
    #     # neurons = []
    #     # for neuron_data in data["neurons"]:
    #     #     neurons.append(
    #     #         Neuron(
    #     #             threshold=neuron_data["threshold"],
    #     #             f_times=neuron_data["f_times"],
    #     #         )
    #     #     )

    #     neurons = [
    #         Neuron(threshold=neuron_data["threshold"], f_times=neuron_data["f_times"])
    #         for neuron_data in data["neurons"]
    #     ]

    #     sources = np.array(
    #         [neuron_data["sources"] for neuron_data in data["neurons"]], dtype=np.int64
    #     )
    #     delays = np.array(
    #         [neuron_data["delays"] for neuron_data in data["neurons"]], dtype=np.float64
    #     )
    #     weights = np.array(
    #         [neuron_data["weights"] for neuron_data in data["neurons"]],
    #         dtype=np.float64,
    #     )

    #     # # Restore connections
    #     # connections = defaultdict(list)
    #     # for k, conns in data["connections"].items():
    #     #     source_id, target_id = map(int, k.split(","))
    #     #     connections[(source_id, target_id)] = [
    #     #         (conn["delay"], conn["weight"]) for conn in conns
    #     #     ]

    #     return cls(neurons, sources, delays, weights)

    # def save_to_json(self, filepath: str):
    #     """Save the network to a JSON file"""
    #     with open(filepath, "w") as f:
    #         json.dump(self.to_dict(), f, indent=2)

    # @classmethod
    # def load_from_json(cls, filepath: str) -> "Simulator":
    #     """Load a network from a JSON file"""
    #     with open(filepath, "r") as f:
    #         data = json.load(f)
    #     return cls.from_dict(data)
