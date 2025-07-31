import json
from typing import List

import numba
import numpy as np
import numpy.typing as npt
from numba import float64, int64, njit
from numba.experimental import jitclass

from rsnn.sim.neuron import Neuron
from rsnn.sim.utils import floyd_warshall
from rsnn.utils import setup_logging

logger = setup_logging(__name__, console_level="INFO", file_level="INFO")

class Simulator:
    """Simulator for a network of neurons."""
    def __init__(
        self,
        neurons: List[Neuron],
        in_sources: npt.NDArray[np.int64],
        in_delays: npt.NDArray[np.float64],
        in_weights: npt.NDArray[np.float64],
    ):
        """
        Initialize the simulator with a list of neurons and connections between them.
        Each neuron has the same number of incoming connections.

        Args:
            neurons (_type_): List of `Neuron` objects representing the neurons in the network, with length L.
            in_sources (_type_): Array of source neuron IDs for each connection, with shape (L, K).
            in_delays (_type_): Array of delays for each connection, with shape (L, K).
            in_weights (_type_): Array of weights for each connection, with shape (L, K).
        """
        self.neurons: List[Neuron] = neurons

        self.in_sources = in_sources
        self.in_delays = in_delays
        self.in_weights = in_weights


        self.in_min_delays = floyd_warshall(self.in_sources, self.in_delays)

    @property
    def n_neurons(self) -> int:
        """Get the number of neurons in the network.

        Returns:
            int: the number of neurons in the network.
        """
        return len(self.neurons)

    @property
    def n_connections(self) -> int:
        """Get the number of connections in the network.

        Returns:
            int: the number of connections in the network.
        """
        # return sum(len(conns) for conns in self.connections.values())
        return self.in_sources.size

    def run(self, start: float, end: float, std_threshold: float = 0.0):
        time = start
        while time < end:
            time = self.step(time, std_threshold)

    @njit(parallel=True)
    def step(self, tmin: float, std_threshold: float = 0.0) -> np.float64:
        # tmax = np.inf
        # logger.debug(f"New step started.")
        f_times = np.full(self.n_neurons, np.nan, dtype=np.float64)

        for l in numba.prange(self.n_neurons):
            # logger.debug(f"Processing neuron {id}.")
            # np.nanmin(f_times + self.in_min_delays[l], initial=np.inf)
            f_times[l] = self.neurons[l].step(
                tmin, np.nanmin(f_times + self.in_min_delays[l], initial=np.inf)
            )
            # f_time = neuron.step(tmin, tmax) # tmax is based on the delay
            # if f_time is not None:
            #     (src_id, tmax) = (id, f_time)

        # Accept the largest possible number of independent spikes
        f_sources = np.argwhere(
            np.all(
                f_times[:, None]
                <= np.nan_to_num(f_times[None, :], nan=np.inf) + self.in_min_delays,
                axis=1,
            )
        ).flatten()
        f_times = f_times[f_sources]
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



