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
    """Note: synapses are first order connections, i.e., alpha kernels"""

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
    """Generate random synapses for a spiking neural network without connectivity restrictions. Note: synapses are initialized with zero-weights.

    Args:
        n_neurons: Number of neurons in the network.
        n_synapses: Total number of synapses to generate.
        min_delay: Minimum connection delay.
        max_delay: Maximum connection delay.
        rng: Random number generator. If None, uses default_rng().

    Returns:
        DataFrame with columns: source, target, delay, w0, and w1.
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
    """Generate fully connected random synapses where every neuron connects to every other neuron. Note: synapses are initialized with zero-weights.

    Args:
        n_neurons: Number of neurons in the network.
        min_delay: Minimum connection delay.
        max_delay: Maximum connection delay.
        rng: Random number generator. If None, uses default_rng().

    Returns:
        DataFrame with columns: source, target, delay, w0, and w1.
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
    """Generate random synapses where each neuron has the same number of incoming connections. Note: synapses are initialized with zero-weights.

    Args:
        n_neurons: Number of neurons in the network.
        n_synapses: Total number of synapses. Must be divisible by n_neurons.
        min_delay: Minimum connection delay.
        max_delay: Maximum connection delay.
        rng: Random number generator. If None, uses default_rng().

    Returns:
        DataFrame with columns: source, target, delay, w0, and w1.

    Raises:
        ValueError: If n_synapses is not divisible by n_neurons.
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
    """Generate random synapses where each neuron has the same number of outgoing connections. Note: synapses are initialized with zero-weights.

    Args:
        n_neurons: Number of neurons in the network.
        n_synapses: Total number of synapses. Must be divisible by n_neurons.
        min_delay: Minimum connection delay.
        max_delay: Maximum connection delay.
        rng: Random number generator. If None, uses default_rng().

    Returns:
        DataFrame with columns: source, target, delay, w0, and w1.

    Raises:
        ValueError: If n_synapses is not divisible by n_neurons.
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
    """Generate random synapses where each neuron has equal incoming and outgoing connections. Note: synapses are initialized with zero-weights.

    Args:
        n_neurons: Number of neurons in the network.
        n_synapses: Total number of synapses. Must be divisible by n_neurons.
        min_delay: Minimum connection delay.
        max_delay: Maximum connection delay.
        rng: Random number generator. If None, uses default_rng().

    Returns:
        DataFrame with columns: source, target, delay, w0, and w1.

    Raises:
        ValueError: If n_synapses is not divisible by n_neurons.
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
