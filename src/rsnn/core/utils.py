import numpy as np
from tqdm.autonotebook import tqdm


def get_stability_matrix(network, spike_train):
    # assumptions: 
    # - causal network
    # - a spike older than spike_train.duration does not influence (directly) the network anymore

    def get_jitter_influence(neuron, ft, neuron_src, ft_src):
        diff = ft - ft_src
        while diff <= 0:
            diff += spike_train.duration
            
        res = 0.0
        if neuron is neuron_src:
            res += neuron.refractory_resp_deriv((diff))
        
        for k in range(neuron.num_synapses):
            if neuron.sources[k] is neuron_src:
                res += neuron.weights[k] * neuron.impulse_resp_deriv((diff - neuron.delays[k]))

        return res

    # def get_index(neuron_idx, firing_time):
    #     # problem to get indices here
    #     return np.argwhere(np.isclose(spike_train.firing_times[neuron_idx], firing_time % spike_train.duration)) + cum_num_firing_times[neuron_idx]

    # cum_num_firing_times = np.cumsum([0] + [spike_train.num_spikes(neuron.idx) for neuron in network.neurons])
    
    num_spikes = spike_train.num_spikes()
    if np.unique(np.concatenate(spike_train.firing_times)).size != num_spikes:
        raise ValueError("Spike train contains non-unique spikes")
    
    # needs joint arrays firing_times / neuron
    neuron_indices = np.concatenate([np.full(spike_train.num_spikes(neuron.idx), neuron.idx) for neuron in network.neurons])
    firing_times = np.concatenate(spike_train.firing_times)
    sorted_indices = np.argsort(firing_times)
    # sorted_firing_times = firing_times[sorted_indices]
    # sorted_neuron_indices = neuron_indices[sorted_indices]

    # dict_firing_times = {}
    # for ft in firing_times:
    #     dict_firing_times[ft] = [neuron for neuron in network.neurons if ft in spike_train.firing_times[neuron.idx]]

    Phi = np.identity(num_spikes)
    for i in tqdm(sorted_indices):
        A = np.identity(num_spikes)

        for j in sorted_indices:
            A[i, j] = get_jitter_influence(
                network.neurons[neuron_indices[i]], 
                firing_times[i],
                network.neurons[neuron_indices[j]], 
                firing_times[j]
                )
        #print(A[get_index(neuron.idx, ft)].min(), A[get_index(neuron.idx, ft)].max())
        A[i] /= A[i].sum()
        Phi = A @ Phi

    return Phi