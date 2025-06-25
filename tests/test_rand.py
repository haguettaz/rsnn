from rsnn.rand import *


def is_valid_f_times(f_times: np.ndarray, period: float=np.inf) -> bool:
    """
    Returns a boolean indicating whether the firing times satisfy the refractory condition.

    Args:
        f_times (np.ndarray): the firing_times
        period (float): the cycle period. For non-periodic spike trains, use np.inf (default).

    Returns:
        (bool): the boolean indicating satisfaction of the refractory condition.
    """
    if f_times.size > 1:
        sorted_f_times = np.sort(f_times)
        isi = np.diff(sorted_f_times, append=sorted_f_times[0] + period)
        return bool(np.all(isi >= REFRACTORY_PERIOD))
    elif f_times.size == 1:
        return period > REFRACTORY_PERIOD
    else:
        return True


def n_incoming(
    id: int, connections: Dict[Tuple[int, int], List[Tuple[float, float]]]
) -> int:
    """
    Returns the number of incoming connections for a given neuron ID.

    Args:
        id (int): The ID of the neuron.
        connections (Dict[Tuple[int, int], List[Tuple[float, float]]]): The connections dictionary.

    Returns:
        int: The number of incoming connections for the neuron.
    """
    return sum(len(conns) for (_, tgt_id), conns in connections.items() if tgt_id == id)


def n_outgoing(
    id: int, connections: Dict[Tuple[int, int], List[Tuple[float, float]]]
) -> int:
    """
    Returns the number of outgoing connections for a given neuron ID.

    Args:
        id (int): The ID of the neuron.
        connections (Dict[Tuple[int, int], List[Tuple[float, float]]]): The connections dictionary.

    Returns:
        int: The number of outgoing connections for the neuron.
    """
    return sum(len(conns) for (src_id, _), conns in connections.items() if src_id == id)


def test_rand_connections_fin():
    """
    Test the random connections function with fixed number of incoming connections.
    """
    n_neurons = 100
    dmin, dmax = 0.0, 1.0

    n_in_per_neuron = 10
    connections = rand_connections_fin(n_neurons, n_in_per_neuron, dmin, dmax)
    print(connections)
    for i in range(n_neurons):
        assert n_incoming(i, connections) == n_in_per_neuron
    for conn in connections:
        assert 0 <= conn[0] < n_neurons and 0 <= conn[1] < n_neurons
        for t, _ in connections[conn]:
            assert dmin <= t <= dmax


def test_rand_connections_fout():
    """
    Test the random connections function with fixed number of outgoing connections.
    """
    n_neurons = 100
    dmin, dmax = 0.0, 1.0

    n_out_per_neuron = 10
    connections = rand_connections_fout(n_neurons, n_out_per_neuron, dmin, dmax)
    for i in range(n_neurons):
        assert n_outgoing(i, connections) == n_out_per_neuron
    for conn in connections:
        assert 0 <= conn[0] < n_neurons and 0 <= conn[1] < n_neurons
        for t, _ in connections[conn]:
            assert dmin <= t <= dmax


def test_rand_connections_fin_fout():
    """
    Test the random connections function with fixed number of incoming and outgoing connections.
    """
    n_neurons = 100
    dmin, dmax = 0.0, 1.0

    n_in_out_per_neuron = 10
    connections = rand_connections_fin_fout(n_neurons, n_in_out_per_neuron, dmin, dmax)
    for i in range(n_neurons):
        assert n_incoming(i, connections) == n_in_out_per_neuron
        assert n_outgoing(i, connections) == n_in_out_per_neuron
    for conn in connections:
        assert 0 <= conn[0] < n_neurons and 0 <= conn[1] < n_neurons
        for t, _ in connections[conn]:
            assert dmin <= t <= dmax


def test_rand_p_f_times():
    """
    Test the random (periodic) firing times generation function.
    """
    n_neurons = 100
    period = 100.0

    f_rate = 0.0
    p_f_times = rand_p_f_times(n_neurons, period, f_rate)
    assert len(p_f_times) == n_neurons
    for p_f_time_n in p_f_times:
        assert is_valid_f_times(p_f_time_n, period)

    f_rate = 1.0
    p_f_times = rand_p_f_times(n_neurons, period, f_rate)
    assert len(p_f_times) == n_neurons
    for p_f_time_n in p_f_times:
        assert is_valid_f_times(p_f_time_n, period)

    f_rate = 10.0
    p_f_times = rand_p_f_times(n_neurons, period, f_rate)
    assert len(p_f_times) == n_neurons
    for p_f_time_n in p_f_times:
        assert is_valid_f_times(p_f_time_n, period)


def test_rand_jit_f_times():
    """
    Test the random jittered firing times generation function.
    """
    f_times = rand_p_f_times(1, 50.0, 1.0)[0]

    jit_f_times = rand_jit_f_times(f_times, 1e-1, -np.inf, np.inf)[0]
    assert is_valid_f_times(jit_f_times)

    jit_f_times = rand_jit_f_times(f_times, 1e-1, 0.0, 50.0)[0]
    assert is_valid_f_times(jit_f_times)
    assert np.all(jit_f_times >= 0.0)
    assert np.all(jit_f_times <= 50.0)

