# import os

# from rsnn.core.network import (Connection, InConnection, Network,
#                                OutConnection, Spike)
# from rsnn.core.neuron import FIRING_THRESHOLD, REFRACTORY_RESET, Neuron, State


# def test_network_initialization():
#     # Create a simple network with 3 neurons
#     network = Network([Neuron() for _ in range(3)])

#     # Check if the neurons are added correctly
#     assert len(network.neurons) == 3


# def test_network_connections():
#     # Create a network with 3 neurons
#     network = Network([Neuron() for _ in range(3)])

#     # Add connections between neurons
#     network.add_connection(0, 1, 1.0, 1.0)
#     network.add_connection(1, 2, 1.0, 1.0)
#     network.add_connection(0, 2, 2.0, 3.0)
#     network.add_connection(0, 1, -1.0, 2.0)

#     # Check if the connections are added correctly
#     assert len(network.outgoing[0]) == 3
#     assert len(network.outgoing[1]) == 1
#     assert len(network.outgoing[2]) == 0
#     assert len(network.incoming[0]) == 0
#     assert len(network.incoming[1]) == 2
#     assert len(network.incoming[2]) == 2
#     assert len(network.connections[(0, 1)]) == 2
#     assert len(network.connections[(0, 2)]) == 1
#     assert len(network.connections[(1, 2)]) == 1
#     assert len(network.connections[(1, 0)]) == 0

#     assert network.get_outgoing(0) == [OutConnection(0, 1, 1.0, 1.0), OutConnection(2, 2, 2.0, 3.0), OutConnection(3, 1, -1.0, 2.0)]
#     assert network.get_incoming(1) == [InConnection(0, 0, 1.0, 1.0), InConnection(3, 0, -1.0, 2.0)]
#     assert network.get_connections_between(0, 1) == [Connection(0, 1.0, 1.0),Connection(3, -1.0, 2.0)]
#     assert network.get_connections_between(1, 2) == [Connection(1, 1.0, 1.0)]
#     assert network.get_connections_between(0, 2) == [Connection(2, 2.0, 3.0)]
#     assert network.get_connections_between(2, 0) == []


# def test_network_add_neuron():
#     # Create a network with 3 neurons
#     network = Network([Neuron() for _ in range(3)])

#     # Add a new neuron to the network
#     new_neuron = Neuron()
#     network.add_neuron(new_neuron)

#     # Check if the neuron is added correctly
#     assert len(network.neurons) == 4
#     assert new_neuron in network.neurons


# def test_network_add_connection():
#     # Create a network with 3 neurons
#     network = Network([Neuron() for _ in range(3)])

#     # Add a connection between two neurons
#     network.add_connection(0, 1, 1.0, 1.0)

#     # Check if the connection is added correctly
#     assert len(network.outgoing[0]) == 1
#     assert len(network.incoming[1]) == 1
#     assert (0, 1) in network.connections
#     assert len(network.connections[(0, 1)]) == 1

#     # Check if the connection parameters are correct
#     assert network.get_connections_between(0, 1) == [Connection(0, 1.0, 1.0)]


# def test_network_update_connectivity():
#     # Create a network with 3 neurons
#     network = Network([Neuron() for _ in range(3)])

#     network.incoming = {
#         0: [],
#         1: [InConnection(0, 0, 1.0, 1.0), InConnection(1, 0, -1.0, 2.0)],
#         2: [
#             InConnection(2, 0, 2.0, 3.0),
#             InConnection(3, 1, 1.0, 1.0),
#             InConnection(4, 1, -0.25, 2.0),
#         ],
#     }
#     assert len(network.incoming[0]) == 0
#     assert len(network.incoming[1]) == 2
#     assert len(network.incoming[2]) == 3

#     network.update_connectivity_from_incoming()

#     assert len(network.outgoing[0]) == 3
#     assert len(network.outgoing[1]) == 2
#     assert len(network.outgoing[2]) == 0

#     assert len(network.connections[(0, 1)]) == 2
#     assert len(network.connections[(0, 2)]) == 1
#     assert len(network.connections[(1, 2)]) == 2

# def test_network_update_fpaths():
#     # Create a network with 3 neurons
#     network = Network([Neuron() for _ in range(3)])

#     # Initialize fastest paths
#     network.update_fpaths()

#     # Check if the fastest paths are initialized correctly
#     assert network.fpaths[(0, 0)] == 0.0
#     assert network.fpaths[(0, 1)] == float("inf")
#     assert network.fpaths[(0, 2)] == float("inf")
#     assert network.fpaths[(0, 3)] == float("inf")
#     assert network.fpaths[(1, 1)] == 0.0
#     assert network.fpaths[(1, 2)] == float("inf")
#     assert network.fpaths[(2, 2)] == 0.0
#     assert network.fpaths[(3, 2)] == float("inf")

#     # Add connections between neurons
#     network.add_connection(0, 1, 1.0, 1.0)
#     network.add_connection(0, 1, -1.0, 2.0)
#     network.add_connection(1, 2, 1.0, 1.0)
#     network.add_connection(0, 2, 2.0, 3.0)

#     # Initialize fastest paths
#     network.update_fpaths()

#     # Check if the fastest paths are initialized correctly
#     assert network.fpaths[(0, 0)] == 0.0
#     assert network.fpaths[(0, 1)] == 1.0
#     assert network.fpaths[(2, 1)] == float("inf")
#     assert network.fpaths[(0, 2)] == 2.0
#     assert network.fpaths[(2, 2)] == 0.0


# def test_network_filter_spikes():
#     # Create a network with 3 neurons
#     network = Network([Neuron() for _ in range(3)])

#     # Add connections between neurons
#     network.add_connection(0, 1, 1.0, 1.0)
#     network.add_connection(0, 1, -1.0, 2.0)
#     network.add_connection(1, 2, 1.0, 1.0)
#     network.add_connection(0, 2, 2.0, 3.0)

#     network.update_fpaths()

#     # Create a list of spikes
#     spikes = [Spike(2, 0.5), Spike(1, 1.5), Spike(0, 0.0), Spike(2, 1.0)]
#     network.filter_and_sort_spikes(spikes)

#     # Check if the spikes are filtered correctly
#     assert len(spikes) == 2
#     assert spikes[0].time == 0.0
#     assert spikes[1].time == 0.5


# def test_network_propagate_spikes():
#     # Create a network with 3 neurons
#     network = Network([Neuron() for _ in range(3)])

#     # Add connections between neurons
#     network.add_connection(0, 1, 1.0, 1.0)
#     network.add_connection(0, 1, -1.0, 2.0)
#     network.add_connection(1, 2, 1.0, 1.0)
#     network.add_connection(0, 2, 2.0, 3.0)

#     spikes = [Spike(0, 0.0), Spike(2, 0.5)]
#     network.propagate_spikes(spikes)

#     # Check if the spikes are propagated correctly
#     # Neuron 0 should have fired at time 0.0 and received no spikes
#     assert len(network.neurons[0].f_times) == 1
#     assert network.neurons[0].f_times[0] == 0.0
#     assert len(network.neurons[0].states) == 1

#     # Neuron 1 should have received two spikes from Neuron 0 (at times 1.0 and 2.0)
#     assert len(network.neurons[1].f_times) == 0
#     assert len(network.neurons[1].states) == 2

#     # Neuron 2 should have fired a time 0.5 and received one spike from Neuron 0 (at time 3.0)
#     assert len(network.neurons[2].f_times) == 1
#     assert len(network.neurons[2].states) == 2


# def test_network_step():
#     # Create a network with 3 neurons
#     network = Network([Neuron() for _ in range(3)])

#     # Add connections between neurons
#     network.add_connection(0, 1, 2.0, 1.0)
#     network.add_connection(0, 1, -0.25, 2.0)
#     network.add_connection(1, 2, 1.0, 1.0)
#     network.add_connection(0, 2, 2.0, 3.0)

#     # Initialize fastest paths
#     network.update_fpaths()

#     # Initialize states of the neurons from a collection of spikes
#     spikes = [Spike(0, 0.0), Spike(1, 2.0)]
#     network.propagate_spikes(spikes)

#     assert len(network.neurons[0].states) == 1
#     assert len(network.neurons[1].states) == 3
#     assert len(network.neurons[2].states) == 2

#     # Step the network at time 0.5
#     spike = network.step(0.0, 0.0)

#     # Check if the spike is propagated correctly
#     assert spike is not None
#     assert spike.time == 3.619061286735945
#     assert spike.source_id == 2


# def test_network_save_and_load():
#     network = Network()

#     neuron = Neuron(
#         0.8,
#         states=[
#             State(4.0, 1.0, 0.0, dc0=1.0),
#             State(2.5, 0.25, 0.5, dc1=-0.25),
#             State(2.0, -1.0, 0.0, dc0=-1.0),
#         ],
#         f_times=[0.0, 2.0],
#     )
#     network.add_neuron(neuron)

#     neuron = Neuron(
#         1.2,
#         states=[
#             State(6.0, 1.0, 0.0, dc0=1.0),
#             State(2.0, 0.0, 1.0, dc0=5.0),
#         ],
#         f_times=[5.0],
#     )
#     network.add_neuron(neuron)

#     neuron = Neuron()
#     network.add_neuron(neuron)

#     network.add_connection(0, 1, 1.0, 1.0)
#     network.add_connection(1, 0, 0.5, 2.0)
#     network.add_connection(2, 0, -0.25, 2.0)

#     # Save the network to a file
#     file_path = "test_network.json"
#     network.save_to_json(file_path)

#     loaded_network = Network.load_from_json(file_path)

#     # Check if the loaded network is equal to the original network
#     assert loaded_network.num_neurons == network.num_neurons
#     assert loaded_network.num_connections == network.num_connections
#     for i in range(len(network.neurons)):
#         assert loaded_network.neurons[i].threshold == network.neurons[i].threshold
#         assert loaded_network.neurons[i].states == network.neurons[i].states
#         assert loaded_network.neurons[i].f_times == network.neurons[i].f_times

#     # Clean up the test file
#     os.remove(file_path)
