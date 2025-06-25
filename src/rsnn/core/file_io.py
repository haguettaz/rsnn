# # from itertools import cycle

# import numpy as np

# from rsnn.utils.file_io import load_object_from_file, save_object_to_file

# # from scipy import sparse
# from .network import Connection, Network, Neuron


# def load_network_from_file(path):
#     """Load the network from a file.

#     Args:
#         path (str): the path to the loading location.

#     Raises:
#         FileNotFoundError: If the file does not exist.
#         ValueError: If number of neurons does not match.
#         ValueError: If error loading the file.
#     """
#     network_dict = load_object_from_file(path)

#     neurons = []
#     for idx, neuron_dict in enumerate(network_dict["neurons"]):
        
#         if neuron_dict["idx"] != idx:
#             raise ValueError("The neuron indices are not consistent.")
        
#         neuron = Neuron(idx, neuron_dict["nominal_threshold"])

#         neuron.firing_times = np.array(neuron_dict["firing_times"])
#         neuron.in_memory = neuron_dict["in_memory"]
#         neuron.in_memory_firing_times = [np.array(ft) for ft in neuron_dict["in_memory_firing_times"]]
#         neuron.in_memory_period = neuron_dict["in_memory_period"]
    
#         neurons.append(neuron)
        
#     connections = [Connection(source=neurons[connection_dict["source_idx"]], target=neurons[connection_dict["target_idx"]], weight=connection_dict["weight"], delay=connection_dict["delay"], order=connection_dict["order"], beta=connection_dict["beta"]) for connection_dict in network_dict["connections"]]
    
#     network = Network(neurons=neurons, connections=connections)
#     network.in_memory = network_dict["in_memory"]
#     network.in_memory_period = network_dict["in_memory_period"]
#     network.in_memory_phis = network_dict["in_memory_phis"]
     
#     return network
    
# def save_network_to_file(network, path):
#     save_object_to_file(network.to_dict(), path)

# # def get_phis(network, firing_times, period, tol=1e-9):
# #     if len(firing_times) != network.num_neurons:
# #         raise ValueError("The number of neurons and the number of firing times channels must be equal.")
    
# #     # create a list of tuples (neuron, firing time), sorted by firing times
# #     all_pairs = [(network.neurons[l], ft) for l in range(network.num_neurons) for ft in firing_times[l]]
# #     all_pairs.sort(key=lambda x: x[1])

# #     # determine mmax, by checking the maximum number of spikes a spike can influence in the future, up to some tolerance
# #     mmax = 0
# #     for n, (source_neuron, source_ft) in enumerate(all_pairs):
# #         m = 0
# #         counter = 0
# #         while counter < len(all_pairs):
# #             target_neuron, target_ft = all_pairs[(n + 1 + m)%len(all_pairs)]
# #             target_ft += (m//len(all_pairs)) * period
# #             m += 1

# #             if sum([co.weight * co.input_kernel_prime((target_ft - source_ft - co.delay)) for co in network.get_connections(source_neuron, target_neuron)]) > tol:
# #                 counter = 0
# #             else:
# #                 counter += 1

# #     Phi = np.identity(mmax)
# #     A = sparse.diags(np.ones(mmax-1), offsets=-1, format="lil")
# #     row = np.zeros(mmax)

# #     # extend the list of pairs on the left, adding the required past spikes 
# #     ext_all_pairs = [(neuron, ft + (i//len(all_pairs))*period) for i, (neuron, ft) in zip(range(mmax + len(all_pairs)), cycle(all_pairs))]
# #     for n, (target_neuron, target_ft) in enumerate(ext_all_pairs[mmax:]):
# #         for m, (source_neuron, source_ft) in enumerate(reversed(ext_all_pairs[n:n+mmax])):
# #             row[m] = sum([co.weight * co.input_kernel_prime((target_ft - source_ft - co.delay)) for co in network.get_connections(source_neuron, target_neuron)])

# #         A[0] = row / np.sum(row)
# #         Phi = A @ Phi

# #     return np.linalg.eigvals(Phi)

