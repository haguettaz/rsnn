from typing import Optional

import numpy as np
import numpy.typing as npt
import polars as pl


def new_synapses(
    sources: npt.NDArray[np.intp],
    targets: npt.NDArray[np.intp],
    delays: npt.NDArray[np.float64] | float = 0.0,
    weight: npt.NDArray[np.float64] | float = 0.0,
):
    """Create a new synapses DataFrame with specified connections.

    Constructs a DataFrame representing synaptic connections between neurons
    with alpha kernel dynamics. Each synapse connects a source to a target
    neuron with specified delay and weight parameters.

    Args:
        sources (npt.NDArray[np.intp]): Array of source neuron indices.
        targets (npt.NDArray[np.intp]): Array of target neuron indices.
        delays (npt.NDArray[np.float64] | float, optional): Synaptic delays.
            Can be array or scalar. Defaults to 0.0.
        weight (npt.NDArray[np.float64] | float, optional): Synaptic weights.
            Can be array or scalar. Defaults to 0.0.

    Returns:
        pl.DataFrame: Synapses DataFrame with columns 'source', 'target',
            'delay', 'weight'.

    Notes:
        Synapses are first-order connections implementing alpha kernels.
        All arrays must have compatible lengths for broadcasting.
    """
    return pl.DataFrame(
        data={
            "source": sources,
            "target": targets,
            "delay": delays,
            "weight": weight,
            # "in_coef_0": in_coef_0,
            # "in_coef_1": in_coef_1,
        },
        schema={
            "source": pl.UInt32,
            "target": pl.UInt32,
            "delay": pl.Float64,
            "weight": pl.Float64,
            # "in_coef_0": pl.Float64,
            # "in_coef_1": pl.Float64,
        },
    )


def rand_synapses(
    n_neurons: int,
    n_synapses: int,
    min_delay: float,
    max_delay: float,
    rng: Optional[np.random.Generator] = None,
) -> pl.DataFrame:
    """Generate random synapses for a spiking neural network without connectivity restrictions.

    Creates random synaptic connections between neurons with uniformly distributed
    delays. Source and target neurons are chosen uniformly at random, allowing
    self-connections and multiple connections between the same neuron pair.

    Args:
        n_neurons (int): Number of neurons in the network.
        n_synapses (int): Total number of synapses to generate.
        min_delay (float): Minimum connection delay.
        max_delay (float): Maximum connection delay.
        rng (Optional[np.random.Generator], optional): Random number generator.
            If None, uses default_rng(). Defaults to None.

    Returns:
        pl.DataFrame: Synapses DataFrame with randomly connected neurons,
            zero-initialized weights, and random delays.

    Notes:
        Synapses are initialized with zero weights and must be trained or
        set manually. No connectivity constraints are applied.
    """
    if rng is None:
        rng = np.random.default_rng()

    return new_synapses(
        sources=rng.integers(n_neurons, size=n_synapses, dtype=np.intp),
        targets=rng.integers(n_neurons, size=n_synapses, dtype=np.intp),
        delays=rng.uniform(min_delay, max_delay, size=n_synapses),
    )


def rand_synapses_fc(
    n_neurons: int,
    min_delay: float,
    max_delay: float,
    rng: Optional[np.random.Generator] = None,
) -> pl.DataFrame:
    """Generate fully connected random synapses where every neuron connects to every other neuron.

    Creates a complete graph of synaptic connections including self-connections.
    Each neuron connects to all neurons in the network with random delays.

    Args:
        n_neurons (int): Number of neurons in the network.
        min_delay (float): Minimum connection delay.
        max_delay (float): Maximum connection delay.
        rng (Optional[np.random.Generator], optional): Random number generator.
            If None, uses default_rng(). Defaults to None.

    Returns:
        pl.DataFrame: Fully connected synapses DataFrame with n_neurons²
            connections, zero-initialized weights, and random delays.

    Notes:
        Generates n_neurons² synapses including self-connections.
        Synapses are initialized with zero weights.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_synapses = n_neurons**2

    return new_synapses(
        # sources=np.repeat(np.arange(n_neurons), n_neurons),
        sources=np.arange(n_synapses) // n_neurons,
        targets=np.arange(n_synapses) % n_neurons,
        delays=rng.uniform(min_delay, max_delay, size=n_synapses),
    )


def rand_synapses_fin(
    n_neurons: int,
    n_synapses: int,
    min_delay: float,
    max_delay: float,
    rng: Optional[np.random.Generator] = None,
) -> pl.DataFrame:
    """Generate random synapses with fixed in-degree connectivity.

    Creates synaptic connections where each neuron receives the same number
    of incoming connections. Source neurons are chosen randomly while
    maintaining equal fan-in for all target neurons.

    Args:
        n_neurons (int): Number of neurons in the network.
        n_synapses (int): Total number of synapses. Must be divisible by n_neurons.
        min_delay (float): Minimum connection delay.
        max_delay (float): Maximum connection delay.
        rng (Optional[np.random.Generator], optional): Random number generator.
            If None, uses default_rng(). Defaults to None.

    Returns:
        pl.DataFrame: Synapses DataFrame with fixed in-degree connectivity,
            zero-initialized weights, and random delays.

    Raises:
        ValueError: If n_synapses is not divisible by n_neurons.

    Notes:
        Each neuron receives exactly n_synapses/n_neurons incoming connections.
        Synapses are initialized with zero weights.
    """
    if n_synapses % n_neurons != 0:
        raise ValueError("n_synapses must be divisible by n_neurons")

    if rng is None:
        rng = np.random.default_rng()

    return new_synapses(
        sources=rng.integers(0, n_neurons, size=n_synapses, dtype=np.intp),
        targets=np.arange(n_synapses) % n_neurons,
        delays=rng.uniform(min_delay, max_delay, size=n_synapses),
    )


def rand_synapses_fout(
    n_neurons: int,
    n_synapses: int,
    min_delay: float,
    max_delay: float,
    rng: Optional[np.random.Generator] = None,
) -> pl.DataFrame:
    """Generate random synapses with fixed out-degree connectivity.

    Creates synaptic connections where each neuron makes the same number
    of outgoing connections. Target neurons are chosen randomly while
    maintaining equal fan-out for all source neurons.

    Args:
        n_neurons (int): Number of neurons in the network.
        n_synapses (int): Total number of synapses. Must be divisible by n_neurons.
        min_delay (float): Minimum connection delay.
        max_delay (float): Maximum connection delay.
        rng (Optional[np.random.Generator], optional): Random number generator.
            If None, uses default_rng(). Defaults to None.

    Returns:
        pl.DataFrame: Synapses DataFrame with fixed out-degree connectivity,
            zero-initialized weights, and random delays.

    Raises:
        ValueError: If n_synapses is not divisible by n_neurons.

    Notes:
        Each neuron makes exactly n_synapses/n_neurons outgoing connections.
        Synapses are initialized with zero weights.
    """
    if n_synapses % n_neurons != 0:
        raise ValueError("n_synapses must be divisible by n_neurons")

    if rng is None:
        rng = np.random.default_rng()

    return new_synapses(
        sources=np.arange(n_synapses) % n_neurons,
        targets=rng.integers(0, n_neurons, size=n_synapses, dtype=np.intp),
        delays=rng.uniform(min_delay, max_delay, size=n_synapses),
    )


def rand_synapses_fin_fout(
    n_neurons: int,
    n_synapses: int,
    min_delay: float,
    max_delay: float,
    rng: Optional[np.random.Generator] = None,
) -> pl.DataFrame:
    """Generate random synapses with balanced in-degree and out-degree connectivity.

    Creates synaptic connections where each neuron has equal incoming and
    outgoing connections. Uses permutation to ensure balanced connectivity
    while maintaining randomness in the connection pattern.

    Args:
        n_neurons (int): Number of neurons in the network.
        n_synapses (int): Total number of synapses. Must be divisible by n_neurons.
        min_delay (float): Minimum connection delay.
        max_delay (float): Maximum connection delay.
        rng (Optional[np.random.Generator], optional): Random number generator.
            If None, uses default_rng(). Defaults to None.

    Returns:
        pl.DataFrame: Synapses DataFrame with balanced connectivity,
            zero-initialized weights, and random delays.

    Raises:
        ValueError: If n_synapses is not divisible by n_neurons.

    Notes:
        Each neuron has exactly n_synapses/n_neurons incoming and outgoing
        connections. Uses random permutation to create balanced connections.
        Synapses are initialized with zero weights.
    """
    if n_synapses % n_neurons != 0:
        raise ValueError("n_synapses must be divisible by n_neurons")

    if rng is None:
        rng = np.random.default_rng()

    return new_synapses(
        sources=np.arange(n_synapses) % n_neurons,
        targets=rng.permutation(np.arange(n_synapses) % n_neurons),
        delays=rng.uniform(min_delay, max_delay, size=n_synapses),
    )
