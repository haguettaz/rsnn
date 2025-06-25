import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.special import lambertw

from rsnn.constants import FIRING_THRESHOLD, REFRACTORY_RESET


def first_crossing(
    start: float,
    length: float,
    c0: float,
    c1: float,
    threshold: float,
) -> Optional[np.float64]:

    if c0 < threshold:
        if c1 > 0:
            dt = -(lambertw(-threshold / c1 * np.exp(-c0 / c1), 0)) - c0 / c1
            if np.isreal(dt) and (dt >= 0.0) and (dt < length):
                return np.float64(start + dt.real)
        elif c1 < 0:
            dt = (
                -(
                    lambertw(
                        -threshold / c1 * np.exp(-c0 / c1),
                        -1,
                    )
                )
                - c0 / c1
            )
            if np.isreal(dt) and (dt >= 0.0) and (dt < length):
                return np.float64(start + dt.real)
        elif threshold < 0:
            dt = np.log(c0 / threshold)
            if (dt >= 0.0) and (dt < length):
                return np.float64(start + dt)
        return None

    else:
        return np.float64(start)


@dataclass
class Neuron:
    """Represents a neuron with its states and firing times."""
    # Firing threshold
    threshold: float = FIRING_THRESHOLD

    # Firing times
    f_times: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty((0,), dtype=np.float64))

    # State variables
    starts: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([-np.inf, np.inf], dtype=np.float64))
    lengths: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([np.inf, np.inf], dtype=np.float64))
    c0s: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros((2,), dtype=np.float64))
    c1s: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros((2,), dtype=np.float64))
    dc0s: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros((2,), dtype=np.float64))
    dc1s: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros((2,), dtype=np.float64))

    # def __init__(
    #     self,
    #     threshold: float = FIRING_THRESHOLD,
    #     f_times: Optional[npt.NDArray[np.float64]] = None,
    # ):
    #     self.threshold = threshold
    #     self.f_times = (
    #         np.sort(f_times)
    #         if f_times is not None
    #         else np.empty((0,), dtype=np.float64)
    #     )

    #     self.clear_states()

    def add_f_times(self, f_times: npt.NDArray[np.float64]):
        """Add firing times to the neuron's firing times."""
        self.f_times = np.sort(np.concatenate((self.f_times, f_times)))

    def init_initial_state(self):
        self.starts[0] = -np.inf
        self.lengths[0] = np.inf
        self.c0s[0] = 0.0
        self.c1s[0] = 0.0
        self.dc0s[0] = 0.0
        self.dc1s[0] = 0.0

    def clear_states(self):
        self.starts = np.array([-np.inf, np.inf], dtype=np.float64)
        self.lengths = np.array([np.inf, np.inf], dtype=np.float64)
        self.c0s = np.zeros((2,), dtype=np.float64)
        self.c1s = np.zeros((2,), dtype=np.float64)
        self.dc0s = np.zeros((2,), dtype=np.float64)
        self.dc1s = np.zeros((2,), dtype=np.float64)
        # self.states = np.concatenate((initial_state(), final_state()), axis=0)

    def add_states(self, starts, lengths, c0s, c1s, dc0s, dc1s):
        """Merge new states into the existing states."""

        self.starts = np.concatenate((self.starts, starts), axis=0)
        self.lengths = np.concatenate((self.lengths, lengths), axis=0)
        self.c0s = np.concatenate((self.c0s, c0s), axis=0)
        self.c1s = np.concatenate((self.c1s, c1s), axis=0)
        self.dc0s = np.concatenate((self.dc0s, dc0s), axis=0)
        self.dc1s = np.concatenate((self.dc1s, dc1s), axis=0)

        sorter = np.argsort(self.starts)
        self.starts = self.starts[sorter]
        self.lengths = self.lengths[sorter]
        self.c0s = self.c0s[sorter]
        self.c1s = self.c1s[sorter]
        self.dc0s = self.dc0s[sorter]
        self.dc1s = self.dc1s[sorter]

    def fire(self, f_time: float, noise: float = 0.0):
        # Add the firing time to the list of firing times
        self.f_times = np.append(self.f_times, f_time)

        # Add threshold noise
        self.threshold = FIRING_THRESHOLD + noise

        # Enter refractory period
        self.recover(f_time)

    def recover(self, f_time: float):
        """
        Clear the states and enter the refractory period at the given firing time.
        This method ensures that the neuron's states consist at least of the initial state, the refractory state, and the final state.
        """
        # Clear states and enter refractory period
        ipos = np.searchsorted(self.starts, f_time, side="left")  # always >= 1
        if ipos > 1:
            self.starts = self.starts[(ipos - 2) :]
            self.lengths = self.lengths[(ipos - 2) :]
            self.c0s = self.c0s[(ipos - 2) :]
            self.c1s = self.c1s[(ipos - 2) :]
            self.dc0s = self.dc0s[(ipos - 2) :]
            self.dc1s = self.dc1s[(ipos - 2) :]

            self.init_initial_state()
        else:
            self.starts = np.concatenate((np.array([-np.inf]), self.starts))
            self.lengths = np.concatenate((np.array([np.inf]), self.lengths))
            self.c0s = np.concatenate((np.array([0.0]), self.c0s))
            self.c1s = np.concatenate((np.array([0.0]), self.c1s))
            self.dc0s = np.concatenate((np.array([0.0]), self.dc0s))
            self.dc1s = np.concatenate((np.array([0.0]), self.dc1s))

        self.starts[1] = f_time
        self.lengths[1] = np.inf
        self.c0s[1] = REFRACTORY_RESET
        self.c1s[1] = 0.0
        self.dc0s[1] = REFRACTORY_RESET
        self.dc1s[1] = 0.0

    def clean_states(self, time: float):
        """
        Clean the states by removing those that end before the given time, while ensuring validity of the first state.
        Note 1: the states necessarily contain at least two states: the initial state and the final state.
        Note 2: one should have self.states[0]["start"] <= time

        Args:
            time (float): The time before which states should be removed.
        """
        ipos = np.searchsorted(self.starts, time, side="right") - 1  # always >= 0
        print(ipos)

        if ipos > 0:  # if there are states to clean
            for i in range(1, ipos + 1):

                dt = np.nan_to_num(self.starts[i] - self.starts[i - 1])
                self.c0s[i] = (self.c0s[i - 1] + dt * self.c1s[i - 1]) * np.exp(
                    -dt
                ) + self.dc0s[i]
                self.c1s[i] = self.c1s[i - 1] * np.exp(-dt) + self.dc1s[i]

                # update_state_forward_(
                #     self.states[i],
                #     self.states[i - 1],
                # )

            self.starts = self.starts[ipos - 1 :]
            self.lengths = self.lengths[ipos - 1 :]
            self.c0s = self.c0s[ipos - 1 :]
            self.c1s = self.c1s[ipos - 1 :]
            self.dc0s = self.dc0s[ipos - 1 :]
            self.dc1s = self.dc1s[ipos - 1 :]

            self.init_initial_state()

            self.dc0s[1] = self.c0s[1]
            self.dc1s[1] = self.c1s[1]

            # init_state(self.states[0], -np.inf, 0.0, 0.0)
            # init_state(
            #     self.states[1],
            #     self.states[1]["start"],
            #     self.states[1]["c0"],
            #     self.states[1]["c1"],
            # )

    def next_firing_time(self, tmax: float) -> Optional[np.float64]:
        t = None

        for n in range(1, self.starts.shape[0] - 1):
            if self.starts[n] < tmax:
                self.lengths[n] = self.starts[n + 1] - self.starts[n]
                self.c0s[n] = (
                    self.c0s[n - 1] + np.nan_to_num(self.lengths[n-1]) * self.c1s[n - 1]
                ) * np.exp(-self.lengths[n-1]) + self.dc0s[n]
                self.c1s[n] = self.c1s[n - 1] * np.exp(-self.lengths[n-1]) + self.dc1s[n]

                # update_state_forward_backward_(
                #     self.states[n],
                #     prev_state=self.states[n - 1],
                #     next_state=self.states[n + 1],
                # )

                t = first_crossing(
                    self.starts[n],
                    self.lengths[n],
                    self.c0s[n],
                    self.c1s[n],
                    self.threshold,
                )
                # t = first_crossing(self.states[n], self.threshold)
                if t is not None:
                    break
            else:
                break


        return t if t is not None and t < tmax else None

    def step(self, tmin, tmax) -> Optional[np.float64]:
        self.clean_states(tmin)
        return self.next_firing_time(tmax)


class Simulator:
    def __init__(
        self,
        neurons: List[Neuron],
        connections: Dict[Tuple[int, int], List[Tuple[float, float]]],
    ):
        """

        Args:
            neurons (_type_): _description_
            connections (Dict[Tuple[int, int], List[Tuple[float, float]]]): _description_
        """
        # self.outgoing: Dict[int, List[OutConnection]] = defaultdict(list)
        # self.incoming: Dict[int, List[InConnection]] = defaultdict(list)
        self.neurons: List[Neuron] = neurons
        # self.connections: Dict[Tuple[int, int], List[Tuple[float, float]]] = defaultdict(list)
        self.connections: Dict[Tuple[int, int], List[Tuple[float, float]]] = connections
        # self.fpaths: Dict[Tuple[int, int], float] = defaultdict(lambda: float("inf"))

    @property
    def num_neurons(self) -> int:
        """Get the number of neurons in the network.

        Returns:
            int: the number of neurons in the network.
        """
        return len(self.neurons)

    @property
    def num_connections(self) -> int:
        """Get the number of connections in the network.

        Returns:
            int: the number of connections in the network.
        """
        return sum(len(conns) for conns in self.connections.values())

    def run(self, start: float, end: float, std_threshold: float = 0.0):
        time = start
        while time < end:
            time = self.step(time, std_threshold)

    def step(self, tmin: float, std_threshold: float = 0.0) -> np.float64:
        tmax = np.inf
        for id, neuron in enumerate(self.neurons):
            f_time = neuron.step(tmin, tmax)
            if f_time is not None:
                (src_id, tmax) = (id, f_time)

        if np.isfinite(tmax):
            self.propagate_spikes(tmax, src_id, std_threshold)

        return np.float64(tmax)

    def propagate_spikes(self, f_time, src_id, std_threshold: float = 0.0):
        # Make the neurons fire
        self.neurons[src_id].fire(f_time, np.random.normal(0, std_threshold))

        # Propagate the spikes to the target neurons, by updating their states
        for tgt_id, tgt_neuron in enumerate(self.neurons):
            starts = np.array(
                [f_time + conn[0] for conn in self.connections[(src_id, tgt_id)]]
            )
            dc1s = np.array([conn[1] for conn in self.connections[(src_id, tgt_id)]])
            tgt_neuron.add_states(
                starts,
                np.full_like(starts, np.inf),
                np.zeros_like(starts),
                np.zeros_like(starts),
                np.zeros_like(starts),
                dc1s,
            )

    def init_from_f_times(self):
        """Initialize the neurons' states based on their firing times."""
        for tgt_id, tgt_neuron in enumerate(self.neurons):
            tgt_neuron.clear_states()
            starts = np.array([
                f_time + conn[0]
                for (src_id, scr_neuron) in enumerate(self.neurons)
                for f_time in scr_neuron.f_times
                for conn in self.connections[(src_id, tgt_id)]
            ])

            dc1s = np.array([
                conn[1]
                for (src_id, scr_neuron) in enumerate(self.neurons)
                for _ in scr_neuron.f_times
                for conn in self.connections[(src_id, tgt_id)]
            ])

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

    def to_dict(self) -> dict:
        """Convert the network to a dictionary for JSON serialization"""
        return {
            "neurons": [
                {
                    "threshold": neuron.threshold,
                    "f_times": neuron.f_times.tolist(),
                }
                for neuron in self.neurons
            ],
            "connections": {
                f"{source_id},{target_id}": [
                    {
                        "delay": conn[0],
                        "weight": conn[1],
                    }
                    for conn in conns
                ]
                for (source_id, target_id), conns in self.connections.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Simulator":
        """Create a network from a dictionary loaded from JSON"""
        # Restore neurons
        neurons = []
        for neuron_data in data["neurons"]:
            neurons.append(
                Neuron(
                    threshold=neuron_data["threshold"],
                    f_times=neuron_data["f_times"],
                )
            )

        # Restore connections
        connections = defaultdict(list)
        for k, conns in data["connections"].items():
            source_id, target_id = map(int, k.split(","))
            connections[(source_id, target_id)] = [
                (conn["delay"], conn["weight"]) for conn in conns
            ]

        return cls(neurons, connections)

    def save_to_json(self, filepath: str):
        """Save the network to a JSON file"""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str) -> "Simulator":
        """Load a network from a JSON file"""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
