import os

from rsnn.constants import FIRING_THRESHOLD, REFRACTORY_RESET
from rsnn.sim import *


def test_neuron_initialization():
    # Create a neuron with default parameters
    neuron = Neuron()

    # Check if the neuron is initialized correctly
    assert neuron.threshold == 1.0
    assert neuron.f_times.size == 0
    assert neuron.starts.size == 2


def test_neuron_merge_states():
    # Create a neuron with default parameters
    neuron = Neuron()

    neuron.add_states(
        np.array([2.0, 0.5, 1.0], dtype=np.float64),
        np.array([np.inf, np.inf, np.inf], dtype=np.float64),
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([2.0, 1.0, -0.25], dtype=np.float64),
    )

    # Check if the states are merged correctly
    assert neuron.starts.size == 5
    assert np.isneginf(neuron.starts[0])
    assert np.isposinf(neuron.starts[-1])
    assert np.allclose(neuron.starts[1:-1], [0.5, 1.0, 2.0])

    # Add more states to the neuron
    neuron.add_states(
        np.array([1.0, 1.5, 4.0, 0.0], dtype=np.float64),
        np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float64),
        np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        np.array([2.0, 1.0, -0.25, -0.25], dtype=np.float64),
    )

    # Check if the states are merged correctly
    assert neuron.starts.size == 9
    assert np.isneginf(neuron.starts[0])
    assert np.isposinf(neuron.starts[-1])
    assert np.allclose(neuron.starts[1:-1], [0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 4.0])


def test_neuron_fire():
    # Create a neuron with default parameters
    neuron = Neuron()

    neuron.add_states(
        np.array([2.0, 0.5, 0.0], dtype=np.float64),
        np.array([np.inf, np.inf, np.inf], dtype=np.float64),
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([2.0, 1.0, -0.25], dtype=np.float64),
    )
    # neuron.add_states(states)

    # Makes the neuron fire again at time 1.0
    neuron.fire(1.0)
    assert neuron.f_times.size == 1
    assert neuron.starts.size == 4
    assert np.isneginf(neuron.starts[0])
    assert np.isposinf(neuron.starts[-1])
    assert np.isclose(neuron.starts[1], 1.0)
    assert np.isclose(neuron.c0s[1], REFRACTORY_RESET)
    assert np.isclose(neuron.c1s[1], 0.0)
    assert np.isclose(neuron.dc0s[1], REFRACTORY_RESET)
    assert np.isclose(neuron.dc1s[1], 0.0)

    # Makes the neuron fire again at time 2.0 and check its fields
    neuron.fire(2.0)
    assert neuron.f_times.size == 2
    assert neuron.starts.size == 4
    assert np.isneginf(neuron.starts[0])
    assert np.isposinf(neuron.starts[-1])
    assert np.isclose(neuron.starts[1], 2.0)
    assert np.isclose(neuron.c0s[1], REFRACTORY_RESET)
    assert np.isclose(neuron.c1s[1], 0.0)
    assert np.isclose(neuron.dc0s[1], REFRACTORY_RESET)
    assert np.isclose(neuron.dc1s[1], 0.0)

    # Makes the neuron fire again at time 3.0 and check its fields
    neuron.fire(3.0)
    assert neuron.f_times.size == 3
    assert neuron.starts.size == 3
    assert np.isneginf(neuron.starts[0])
    assert np.isposinf(neuron.starts[-1])
    assert np.isclose(neuron.starts[1], 3.0)
    assert np.isclose(neuron.c0s[1], REFRACTORY_RESET)
    assert np.isclose(neuron.c1s[1], 0.0)
    assert np.isclose(neuron.dc0s[1], REFRACTORY_RESET)
    assert np.isclose(neuron.dc1s[1], 0.0)


def test_neuron_clean_states():
    # Create a neuron with default parameters
    neuron = Neuron()

    neuron.add_states(
        np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float64),
        np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float64),
        np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
    )

    # Clean the states at time 1.5
    neuron.clean_states(1.5)
    assert len(neuron.starts) == 5
    assert np.isneginf(neuron.starts[0])
    assert np.isposinf(neuron.starts[-1])
    assert np.allclose(neuron.starts[1:-1], [1.0, 2.0, 4.0])
    assert np.isclose(neuron.c0s[1], 0.3032653298563167)
    assert np.isclose(neuron.c1s[1], 1.6065306597126334)

    # Clean the states at time 2.0
    neuron.clean_states(2.0)
    assert len(neuron.starts) == 4
    assert np.isneginf(neuron.starts[0])
    assert np.isposinf(neuron.starts[-1])
    assert np.allclose(neuron.starts[1:-1], [2.0, 4.0])
    assert np.isclose(neuron.c0s[1], 0.7025746813940871)
    assert np.isclose(neuron.c1s[1], 1.5910096013198722)

    # Clean the states at time 5.0
    neuron.clean_states(5.0)
    assert len(neuron.starts) == 3
    assert np.isneginf(neuron.starts[0])
    assert np.isposinf(neuron.starts[-1])
    assert np.allclose(neuron.starts[1:-1], [4.0])
    assert np.isclose(neuron.c0s[1], 0.525722613554932)
    assert np.isclose(neuron.c1s[1], 1.2153197350267952)


def test_neuron_next_firing_time():
    # Create a neuron with default parameters
    neuron = Neuron()

    neuron.add_states(
        np.array([0.0, 0.5, 1.0], dtype=np.float64),
        np.array([np.inf, np.inf, np.inf], dtype=np.float64),
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([2.0, 1.0, -0.25], dtype=np.float64),
    )

    # Get the next firing time
    next_time = neuron.next_firing_time(np.inf)
    assert next_time is not None and np.isclose(next_time, 0.9000963859659488)
    next_time = neuron.next_firing_time(0.5)
    assert next_time is None

    # Add firing time to the neuron, resetting the neuron potential
    neuron.fire(0.75)
    next_time = neuron.next_firing_time(np.inf)
    assert next_time is None


def test_neuron_step():
    # Create a neuron with default parameters
    neuron = Neuron()

    neuron.add_states(
        np.array([0.0, 0.5, 1.0], dtype=np.float64),
        np.array([np.inf, np.inf, np.inf], dtype=np.float64),
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([2.0, 1.0, -0.25], dtype=np.float64),
    )
    # neuron.add_states(states)

    # Step the neuron at time 0.0
    next_time = neuron.step(0.0, np.inf)
    assert neuron.starts.size == 5
    assert next_time is not None and np.isclose(next_time, 0.9000963859659488)

    # Step the neuron at time 0.75
    next_time = neuron.step(0.75, np.inf)
    assert neuron.starts.size == 4
    assert next_time is not None and np.isclose(next_time, 0.9000963859659488)


def test_simulator_initialization():
    rng = np.random.default_rng(42)

    n_neurons = 3
    n_inputs = 5 # number of inputs per neuron

    # Create a list of neurons
    neurons = [Neuron() for _ in range(n_neurons)]

    # Create connections between neurons
    co_sources = rng.integers(0, n_neurons, size=(n_neurons, n_inputs))
    co_delays = rng.uniform(0.0, 10.0, size=(n_neurons, n_inputs))
    co_weights = np.zeros_like(co_delays)

    simulator = Simulator(neurons, co_sources, co_delays, co_weights)

    # Check if the neurons are added correctly
    assert simulator.n_neurons == n_neurons
    assert simulator.n_connections == n_neurons * n_inputs

def test_simulator_save_and_load():
    rng = np.random.default_rng(42)

    n_neurons = 3
    n_inputs = 5 # number of inputs per neuron

    # Create a list of neurons
    neurons = [
        Neuron(0.5, np.array([0.0, 2.0])),
        Neuron(1.25, np.array([1.0, 3.0])),
        Neuron(1.0, np.array([1.0, 7.0])),
    ]

    # Create connections between neurons
    co_sources = rng.integers(0, n_neurons, size=(n_neurons, n_inputs))
    co_delays = rng.uniform(0.0, 10.0, size=(n_neurons, n_inputs))
    co_weights = np.zeros_like(co_delays)

    # Create a simulator with these neurons and connections
    simulator = Simulator(neurons, co_sources, co_delays, co_weights)

    # Save the simulator to a file
    file_path = "test_simulator.json"
    simulator.save_to_json(file_path)

    # Load the simulator from the file
    loaded_simulator = Simulator.load_from_json(file_path)

    # Check if the loaded simulator is equal to the original simulator
    assert loaded_simulator.n_neurons == simulator.n_neurons
    assert loaded_simulator.n_connections == simulator.n_connections
    for i in range(len(simulator.neurons)):
        assert loaded_simulator.neurons[i].threshold == simulator.neurons[i].threshold
        assert np.array_equal(
            loaded_simulator.neurons[i].f_times, simulator.neurons[i].f_times
        )

    os.remove(file_path)  # Clean up the test file


def test_simulator_propagate_spikes():
    # Create a list of neurons
    neurons = [Neuron() for _ in range(3)]

    # Create connections between neurons
    co_sources = np.array([[1, 2], [0, 0], [2, 0]])
    co_delays = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]])
    co_weights = np.zeros_like(co_delays)

    # Create a simulator with these neurons and connections
    simulator = Simulator(neurons, co_sources, co_delays, co_weights)

    # Neuron 0 fires at time 1.0
    simulator.propagate_spikes(f_time=1.0, src_id=0, std_threshold=1e-2)

    assert np.allclose(simulator.neurons[0].f_times, [1.0])
    assert np.allclose(simulator.neurons[1].f_times, [])
    assert np.allclose(simulator.neurons[2].f_times, [])

    assert np.allclose(simulator.neurons[0].starts[1:-1], [1.0])
    assert np.allclose(simulator.neurons[1].starts[1:-1], [3.0, 4.0])
    assert np.allclose(simulator.neurons[2].starts[1:-1], [2.0])


def test_simulator_init_from_f_times():
    # Create a list of neurons
    neurons: List[Neuron] = [
        Neuron(f_times = np.array([0.0, 1.0])),
        Neuron(f_times = np.array([2.0])),
        Neuron(f_times = np.array([1.0, 4.0])),
    ]

    # Create connections between neurons
    co_sources = np.array([[1, 2], [0, 0], [2, 0]])
    co_delays = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]])
    co_weights = np.zeros_like(co_delays)

    # Create a simulator with these neurons and connections
    simulator = Simulator(neurons, co_sources, co_delays, co_weights)
    simulator.init_from_f_times()

    assert np.allclose(simulator.neurons[0].f_times, [0.0, 1.0])
    assert np.allclose(simulator.neurons[1].f_times, [2.0])
    assert np.allclose(simulator.neurons[2].f_times, [1.0, 4.0])

    assert np.allclose(simulator.neurons[0].starts[1:-1], [1.0, 3.0, 3.0, 6.0])
    assert np.allclose(simulator.neurons[1].starts[1:-1], [2.0, 2.0, 3.0, 3.0, 4.0])
    assert np.allclose(simulator.neurons[2].starts[1:-1], [4.0, 4.0, 7.0])


def test_simulator_step():
    # # Create a list of neurons
    # neurons: List[Neuron] = [
    #     Neuron(f_times=np.array([0.0])),
    #     Neuron(f_times=np.array([2.0])),
    #     Neuron(),
    # ]

    # # Create a dictionary of connections between these neurons
    # connections: Dict[Tuple[int, int], List[Tuple[float, float]]] = defaultdict(list)
    # connections[(0, 1)].append((1.0, 2.0))
    # connections[(0, 1)].append((2.0, -0.25))
    # connections[(1, 2)].append((1.0, 1.0))
    # connections[(0, 2)].append((3.0, 2.0))

    # # Create a simulator with these neurons and connections
    # simulator = Simulator(neurons, connections)
    # simulator.init_from_f_times()

    # Create a list of neurons
    neurons: List[Neuron] = [
        Neuron(f_times = np.array([0.0, 1.0])),
        Neuron(f_times = np.array([2.0])),
        Neuron(f_times = np.array([1.0, 4.0])),
    ]

    # Create connections between neurons
    co_sources = np.array([[1, 2], [0, 0], [2, 0]])
    co_delays = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]])
    co_weights = np.ones_like(co_delays)

    # Create a simulator with these neurons and connections
    simulator = Simulator(neurons, co_sources, co_delays, co_weights)
    simulator.init_from_f_times()

    # Step the simulator at time 0.0
    time = simulator.step(0.0)
    assert time is not None
    assert np.allclose(neurons[0].f_times, [0.0, 1.0])
    assert np.allclose(neurons[1].f_times, [2.0, 4.16437639])
    assert np.allclose(neurons[2].f_times, [1.0, 4.0])
