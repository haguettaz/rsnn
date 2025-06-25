import json
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from rsnn.core.neuron import Neuron, State


@dataclass
class Spike:
    source_id: int
    time: float


@dataclass
class Connection:
    id: int
    weight: float
    delay: float


@dataclass
class OutConnection:
    id: int
    target_id: int
    weight: float
    delay: float


@dataclass
class InConnection:
    id: int
    source_id: int
    weight: float
    delay: float


@dataclass
class Network:
    """
    This class represents a network of neurons, allowing for efficient
    management of connections and propagation of spikes.
    It provides methods to add neurons, manage connections, and simulate
    the network's behavior over time.
    Attributes:
        neurons (List[Neuron]): List of neurons in the network.
        outgoing (Dict[int, List[Tuple[int, float, float]]]): Outgoing connections from each neuron, where the key is the source ID and the value is a list of tuples (weight, delay).
        incoming (Dict[int, List[Tuple[int, float, float]]]): Incoming connections to each neuron, where the key is the target ID and the value is a list of tuples (weight, delay).
        connections (Dict[Tuple[int, int], List[Tuple[float, float]]]): All connections between neurons, where the key is a tuple of (source_id, target_id) and the value is a list of tuples (weight, delay).
        fpaths (Dict[Tuple[int, int], float]): Fastest paths between neurons.
    """

    def __init__(self, neurons: Optional[List[Neuron]] = None):
        self.neurons = neurons if neurons is not None else []

        # self.outgoing: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
        # self.incoming: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
        # self.connections: Dict[Tuple[int, int], List[Tuple[float, float]]] = (
        #     defaultdict(list)
        # )
        self.outgoing: Dict[int, List[OutConnection]] = defaultdict(list)
        self.incoming: Dict[int, List[InConnection]] = defaultdict(list)
        self.connections: Dict[Tuple[int, int], List[Connection]] = defaultdict(list)
        self.fpaths: Dict[Tuple[int, int], float] = defaultdict(lambda: float("inf"))

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

    def add_neuron(self, neuron: Neuron):
        """Add a neuron to the network.

        Args:
            neuron (Neuron): the neuron to be added to the network.
        """
        self.neurons.append(neuron)

    def add_connection(
        self, source_id: int, target_id: int, weight: float, delay: float
    ):
        """Add a connection between neurons"""

        id = self.num_connections
        self.outgoing[source_id].append(OutConnection(id, target_id, weight, delay))
        self.incoming[target_id].append(InConnection(id, source_id, weight, delay))
        self.connections[(source_id, target_id)].append(Connection(id, weight, delay))

    # def get_outgoing(self, source_id: int) -> List[Tuple[int, float, float]]:
    def get_outgoing(self, source_id: int) -> List[OutConnection]:
        """Get all outgoing connections from a neuron - O(1) access"""
        return self.outgoing[source_id]

    # def get_incoming(self, target_id: int) -> List[Tuple[int, float, float]]:
    def get_incoming(self, target_id: int) -> List[InConnection]:
        """Get all incoming connections to a neuron - O(1) access"""
        return self.incoming[target_id]

    # def get_connections_between(
    #     self, source_id: int, target_id: int
    # ) -> List[Tuple[float, float]]:
    def get_connections_between(
        self, source_id: int, target_id: int
    ) -> List[Connection]:
        """Get all connections between two neurons - O(1) access"""
        return self.connections[(source_id, target_id)]

    def update_connectivity_from_incoming(self):
        """Update the network connectivity based on incoming connections."""
        self.connections.clear()
        self.outgoing.clear()
        for target_id, in_conns in self.incoming.items():
            for in_conn in in_conns:
                self.connections[(in_conn.source_id, target_id)].append(
                    Connection(in_conn.id, in_conn.weight, in_conn.delay)
                )
                self.outgoing[in_conn.source_id].append(
                    OutConnection(in_conn.id, target_id, in_conn.weight, in_conn.delay)
                )

        self.update_fpaths()

    def update_connectivity_from_outgoing(self):
        """Update the network connectivity based on outgoing connections."""
        self.connections.clear()
        self.incoming.clear()
        for source_id, out_conns in self.outgoing.items():
            for out_conn in out_conns:
                self.connections[(source_id, out_conn.target_id)].append(
                    Connection(out_conn.id, out_conn.weight, out_conn.delay)
                )
                self.incoming[out_conn.target_id].append(
                    InConnection(
                        out_conn.id, source_id, out_conn.weight, out_conn.delay
                    )
                )

        self.update_fpaths()

    def update_connectivity_from_connections(self):
        """Update the network connectivity based on all connections."""
        self.outgoing.clear()
        self.incoming.clear()
        for (source_id, target_id), conns in self.connections.items():
            for conn in conns:
                self.outgoing[source_id].append(
                    OutConnection(conn.id, target_id, conn.weight, conn.delay)
                )
                self.incoming[target_id].append(
                    InConnection(conn.id, source_id, conn.weight, conn.delay)
                )

        self.update_fpaths()

    def update_fpaths(self):
        """Initialize the fastest paths between neurons, using the Floyd-Warshall algorithm."""
        for source_id in range(len(self.neurons)):
            self.fpaths[(source_id, source_id)] = 0.0
            for target_id in range(source_id):
                self.fpaths[(source_id, target_id)] = min(
                    map(
                        lambda conn: conn.delay,
                        self.get_connections_between(source_id, target_id),
                    ),
                    default=float("inf"),
                )
                self.fpaths[(target_id, source_id)] = min(
                    map(
                        lambda conn: conn.delay,
                        self.get_connections_between(target_id, source_id),
                    ),
                    default=float("inf"),
                )

        for inter_id in range(len(self.neurons)):
            for target_id in range(len(self.neurons)):
                for source_id in range(len(self.neurons)):
                    if (
                        self.fpaths[(source_id, inter_id)]
                        + self.fpaths[(inter_id, target_id)]
                        < self.fpaths[(source_id, target_id)]
                    ):
                        self.fpaths[(source_id, target_id)] = (
                            self.fpaths[(source_id, inter_id)]
                            + self.fpaths[(inter_id, target_id)]
                        )

    def step(self, time: float, std_threshold: float = 0.0) -> Optional[Spike]:
        new_spikes = []
        for neuron_id in range(self.num_neurons):
            firing_time = self.neurons[neuron_id].step(time)
            if firing_time is not None:
                new_spikes.append(Spike(neuron_id, firing_time))
        self.filter_and_sort_spikes(new_spikes)

        self.propagate_spikes(new_spikes, std_threshold)

        return new_spikes[0] if new_spikes else None

    def filter_and_sort_spikes(self, spikes: List[Spike]):
        """
        Retain only the earliest spike and the independent ones.

        Args:
            spikes (List[Spike]): List of spikes to filter.
        """
        spikes.sort(key=lambda s: s.time)
        num_spikes = len(spikes)

        for pos, spike in enumerate(reversed(spikes)):
            if any(
                other_spike.time + self.fpaths[(other_spike.source_id, spike.source_id)]
                < spike.time
                for other_spike in spikes[: num_spikes - pos - 1]
            ):
                spikes.remove(spike)

    def propagate_spikes(self, spikes: List[Spike], std_threshold: float = 0.0):
        print(f"Spikes to propagate: {spikes}")

        # Make the neurons fire
        for spike in spikes:
            self.neurons[spike.source_id].fire(
                spike.time, np.random.normal(0, std_threshold)
            )

        # Propagate the spikes to the target neurons, by updating their states
        for target_id in range(self.num_neurons):
            states = [
                State(spike.time + conn.delay, 0.0, conn.weight, dc1=conn.weight)
                for spike in spikes
                for conn in self.get_connections_between(spike.source_id, target_id)
            ]

            self.neurons[target_id].merge_states(states)

    def sim(self, start: float, end: float, std_threshold: float = 0.0):
        time = start
        while time < end:
            next_spike = self.step(time, std_threshold)
            time = next_spike.time if next_spike else end

    def to_dict(self) -> dict:
        """Convert the network to a dictionary for JSON serialization"""
        return {
            "neurons": [
                {
                    "threshold": neuron.threshold,
                    "states": [
                        {
                            "start": state.start,
                            "c0": state.c0,
                            "c1": state.c1,
                            "dc0": state.dc0,
                            "dc1": state.dc1,
                            "length": (
                                state.length if state.length != float("inf") else None
                            ),
                        }
                        for state in neuron.states
                    ],
                    "f_times": neuron.f_times,
                }
                for neuron in self.neurons
            ],
            "connections": {
                f"{source_id},{target_id}": [
                    {
                        "id": conn.id,
                        "weight": conn.weight,
                        "delay": conn.delay,
                    }
                    for conn in conns
                ]
                for (source_id, target_id), conns in self.connections.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Network":
        """Create a network from a dictionary loaded from JSON"""
        # Create neurons
        neurons = []
        for neuron_data in data["neurons"]:
            states = []
            for state_data in neuron_data["states"]:
                length = (
                    state_data["length"]
                    if state_data["length"] is not None
                    else float("inf")
                )
                state = State(
                    start=state_data["start"],
                    c0=state_data["c0"],
                    c1=state_data["c1"],
                    dc0=state_data["dc0"],
                    dc1=state_data["dc1"],
                    length=length,
                )
                states.append(state)

            neuron = Neuron(
                threshold=neuron_data["threshold"],
                states=states,
                f_times=neuron_data["f_times"],
            )
            neurons.append(neuron)

        # Create network
        network = cls(neurons)

        # Restore connections
        network.connections = defaultdict(list)
        for k, conns in data["connections"].items():
            source_id, target_id = map(int, k.split(","))
            network.connections[(source_id, target_id)] = [
                Connection(id=conn["id"], weight=conn["weight"], delay=conn["delay"])
                for conn in conns
            ]

        network.update_connectivity_from_connections()

        return network

    def save_to_json(self, filepath: str):
        """Save the network to a JSON file"""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str) -> "Network":
        """Load a network from a JSON file"""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
