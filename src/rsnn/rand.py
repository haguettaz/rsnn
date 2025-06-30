from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.stats import truncnorm

from rsnn.constants import REFRACTORY_PERIOD


def build_connections(
    src_ids: npt.NDArray[np.intp],
    tgt_ids: npt.NDArray[np.intp],
    delays: npt.NDArray[np.float64],
) -> Dict[Tuple[int, int], List[Tuple[float, float]]]:
    """Helper function to build connections dictionary from source and target IDs and delays.

    Args:
        src_ids (npt.NDArray[np.intp]): Array of source neuron IDs
        tgt_ids (npt.NDArray[np.intp]): Array of target neuron IDs
        delays (npt.NDArray[np.float64]): Array of connection delays

    Returns:
        Dict[Tuple[int, int], List[Tuple[float, float]]]: Dictionary of connections where keys are tuples
        of (source_id, target_id) and values are lists of (delay, weight) tuples.
    """
    connections: Dict[Tuple[int, int], List[Tuple[float, float]]] = defaultdict(list)
    for i in range(len(src_ids)):
        key = (int(src_ids[i]), int(tgt_ids[i]))
        connections[key].append((float(delays[i]), 0.0))
    return connections


def rand_connections_fin(
    n_neurons: int,
    n_in_per_neuron: int,
    min_delay: float,
    max_delay: float,
    rng: Optional[np.random.Generator] = None,
) -> Dict[Tuple[int, int], List[Tuple[float, float]]]:
    """Generate random connections for a neural network, where each neuron has a fixed number of incoming connections.

    Args:
        n_neurons (int): Number of neurons in the network
        n_in_per_neuron (int): Number of incoming connections per neuron
        min_delay (float): Minimum connection delay
        max_delay (float): Maximum connection delay
        rng (Optional[np.random.Generator]): Random number generator. If None, uses default_rng()

    Returns:
        Dict[Tuple[int, int], List[Tuple[float, float]]]: Dictionary of connections where keys are tuples
        of (source_id, target_id) and values are lists of (delay, weight) tuples.
    """
    if rng is None:
        rng = np.random.default_rng()

    total_connections = n_neurons * n_in_per_neuron

    src_ids: npt.NDArray[np.intp] = rng.integers(
        0, n_neurons, size=total_connections, dtype=np.intp
    )
    tgt_ids: npt.NDArray[np.intp] = np.arange(n_neurons, dtype=np.intp).repeat(
        n_in_per_neuron
    )
    delays: npt.NDArray[np.float64] = rng.uniform(
        min_delay, max_delay, size=total_connections
    )

    return build_connections(src_ids, tgt_ids, delays)


def rand_connections_fout(
    n_neurons: int,
    n_out_per_neuron: int,
    min_delay: float,
    max_delay: float,
    rng: Optional[np.random.Generator] = None,
) -> Dict[Tuple[int, int], List[Tuple[float, float]]]:
    """Generate random connections for a neural network, where each neuron has a fixed number of outgoing connections.

    Args:
        n_neurons (int): Number of neurons in the network
        n_out_per_neuron (int): Number of outgoing connections per neuron
        min_delay (float): Minimum connection delay
        max_delay (float): Maximum connection delay
        rng (Optional[np.random.Generator]): Random number generator. If None, uses default_rng()

    Returns:
        Dict[Tuple[int, int], List[Tuple[float, float]]]: Dictionary of connections where keys are tuples
        of (source_id, target_id) and values are lists of (delay, weight) tuples.
    """
    if rng is None:
        rng = np.random.default_rng()

    total_connections = n_neurons * n_out_per_neuron

    src_ids: npt.NDArray[np.intp] = np.arange(n_neurons, dtype=np.intp).repeat(
        n_out_per_neuron
    )
    tgt_ids: npt.NDArray[np.intp] = rng.integers(
        0, n_neurons, size=total_connections, dtype=np.intp
    )
    delays: npt.NDArray[np.float64] = rng.uniform(
        min_delay, max_delay, size=total_connections
    )

    return build_connections(src_ids, tgt_ids, delays)


def rand_connections_fin_fout(
    n_neurons: int,
    n_in_out_per_neuron: int,
    min_delay: float,
    max_delay: float,
    rng: Optional[np.random.Generator] = None,
) -> Dict[Tuple[int, int], List[Tuple[float, float]]]:
    """Generate random connections for a neural network, where each neuron has a fixed number of outgoing connections.

    Args:
        n_neurons (int): Number of neurons in the network
        n_in_out_per_neuron (int): Number of incoming and outgoing connections per neuron
        min_delay (float): Minimum connection delay
        max_delay (float): Maximum connection delay
        rng (Optional[np.random.Generator]): Random number generator. If None, uses default_rng()

    Returns:
        Dict[Tuple[int, int], List[Tuple[float, float]]]: Dictionary of connections where keys are tuples
        of (source_id, target_id) and values are lists of (delay, weight) tuples.
    """
    if rng is None:
        rng = np.random.default_rng()

    total_connections = n_neurons * n_in_out_per_neuron

    src_ids: npt.NDArray[np.intp] = np.arange(n_neurons, dtype=np.intp).repeat(
        n_in_out_per_neuron
    )
    np.random.shuffle(src_ids)
    tgt_ids: npt.NDArray[np.intp] = np.arange(n_neurons, dtype=np.intp).repeat(
        n_in_out_per_neuron
    )
    np.random.shuffle(tgt_ids)
    delays: npt.NDArray[np.float64] = rng.uniform(
        min_delay, max_delay, size=total_connections
    )

    return build_connections(src_ids, tgt_ids, delays)


def pmf_n_f_times(
    period: float, f_rate: float
) -> Tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]:
    """
    Returns the probability mass function of the number of spikes in a periodic spike train with a given period and firing rate. Note: for numerical stability, the pmf is computed first in the log domain.

    Args:
        period (float): the period of the spike train in [tau_0].
        f_rate (float): the firing rate of the spike train in [1/tau_0].

    Returns:
        Tuple[np.ndarray, np.ndarray]: the support of the pmf and the pmf.
    """
    ns = np.arange(period, dtype=int)
    logpns = (ns - 1) * np.log(period - ns) + ns * np.log(f_rate)
    logpns[1:] -= np.cumsum(np.log(ns[1:]))
    logpns -= np.max(logpns)  # to avoid overflow when exponentiating
    pns = np.exp(logpns)
    return ns, pns / np.sum(pns)


def expected_n_f_times(period: float, f_rate: float) -> float:
    """
    Returns the expected number of spikes in a periodic spike train with a given period and firing rate.

    Args:
        period (float): the period of the spike train in [tau_0].
        f_rate (float): the firing rate of the spike train in [1/tau_0].

    Returns:
        float: the expected number of spikes.
    """
    ns, pns = pmf_n_f_times(period, f_rate)
    return np.inner(ns, pns)


def rand_f_times(
    n_channels: int,
    period: float,
    f_rate: float,
    rng: Optional[np.random.Generator] = None,
) -> List[npt.NDArray[np.float64]]:
    """
    Returns a random multi-channel periodic spike train.

    Args:
        period (float): The cycle period of the spike train.
        f_rate (float): The firing rate of the spike train.
        n_channels (int): The number of channels / neurons.
        rng (np.random.Generator, optional): The random number generator. If None, uses default_rng()

    Raises:
        ValueError: If the period is negative.
        ValueError: If the firing rate is negative.

    Returns:
        (List[npt.NDArray[np.float64]]): a multi-channel periodic spike train.
    """
    if period < 0.0:
        raise ValueError(f"The period should be non-negative.")

    if f_rate < 0.0:
        raise ValueError(f"The firing rate should be non-negative.")

    if period <= REFRACTORY_PERIOD or f_rate == 0.0:
        return [np.array([])] * n_channels

    if rng is None:
        rng = np.random.default_rng()

    multi_f_times = []

    ns, pns = pmf_n_f_times(period, f_rate)

    for _ in range(n_channels):
        # Sample the number of spikes in [0, period)
        n = rng.choice(ns, p=pns)
        if n > 0:
            # sample the effective poisson process in [0, period-n)
            f_times = np.full(n, rng.uniform(0, period))
            f_times[1:] += np.sort(rng.uniform(0, period - n, n - 1)) + np.arange(1, n)

            # transform the effective poisson process into a periodic spike train ...
            multi_f_times.append(f_times % period)
        else:
            multi_f_times.append(np.array([]))

    return multi_f_times


def rand_jit_f_times(
    f_times: npt.NDArray[np.float64],
    std_jitter: float,
    start: float = -np.inf,
    end: float = np.inf,
    n_iter: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> npt.NDArray[np.float64]:
    """
    Returns a (Gaussian) jittered version of the given spike train.
    It uses the Gibbs sampler.
    A sorted copy of the firing times is jittered iteratively, alternating between even and odd indices to ensure that the refractory period is respected everywhere.

    Args:
        spike_train (npt.NDArray[np.float64]): the nominal firing locations
        std_jitter (float): the standard deviation of the Gaussian jitter noise
        tmin (float, optional): the lower bound of the time range. Defaults to -np.inf
        tmax (float, optional): the upper bound of the time range. Defaults to np.inf
        n_iter (int, optional): the maximum number of iterations. Defaults to 1000
        rng (np.random.Generator, optional): the random number generator. If None, uses default_rng()

    Raises:
        ValueError: If the firing times are not in the range [start, end].


    Returns:
        npt.NDArray[np.float64]: the jittered spike train
    """
    if f_times.min(initial=np.inf) < start or f_times.max(initial=-np.inf) > end:
        raise ValueError(
            f"The firing times should be in the range [{start}, {end}]. "
            f"Got [{f_times.min()}, {f_times.max()}]."
        )
    
    sampler = lambda a_, b_, loc_: truncnorm.rvs(
        (a_ - loc_) / std_jitter,
        (b_ - loc_) / std_jitter,
        loc=loc_,
        scale=std_jitter,
        random_state=rng or np.random.default_rng(),
    )

    jit_f_times = np.sort(f_times) # make a sorted copy of the firing times

    if jit_f_times.size > 1:
        tmin = np.full(jit_f_times.size, start)
        tmax = np.full(jit_f_times.size, end)

        even = np.arange(0, jit_f_times.size, 2)
        odd = np.arange(1, jit_f_times.size, 2)

        for _ in range(n_iter):
            # fix odd indices and sample the even ones
            tmin[1:] = jit_f_times[:-1] + REFRACTORY_PERIOD
            tmax[:-1] = jit_f_times[1:] - REFRACTORY_PERIOD
            print(f"Sampling even indices around {jit_f_times[even]} within {tmin[even]} -- {tmax[even]}")
            jit_f_times[even] = sampler(
                tmin[even],
                tmax[even],
                f_times[even],
            )

            # fix even indices and sample odd ones
            tmin[1:] = jit_f_times[:-1] + REFRACTORY_PERIOD
            tmax[:-1] = jit_f_times[1:] - REFRACTORY_PERIOD
            print(f"Sampling odd indices around {jit_f_times[odd]} within {tmin[odd]} -- {tmax[odd]}")
            jit_f_times[odd] = sampler(
                tmin[odd],
                tmax[odd],
                f_times[odd],
            )

    elif jit_f_times.size == 1:
        jit_f_times[0] = sampler(start, end, f_times[0])

    return jit_f_times
