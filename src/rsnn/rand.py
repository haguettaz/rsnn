from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.stats import truncnorm

from .channels import new_channels
from .constants import REFRACTORY_PERIOD


def rand_synapses(
    n_neurons: int,
    n_synapses: int,
    min_delay: float,
    max_delay: float,
    rng: Optional[np.random.Generator] = None,
) -> pl.DataFrame:
    """Generate random synapses for a spiking neural network without connectivity restrictions.

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

    return new_channels(
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

    return new_channels(
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
    """Generate random synapses where each neuron has the same number of incoming connections.

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

    return new_channels(
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
    """Generate random synapses where each neuron has the same number of outgoing connections.

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

    return new_channels(
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
    """Generate random synapses where each neuron has equal incoming and outgoing connections.

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

    return new_channels(
        sources=np.arange(n_synapses) % n_neurons,
        targets=rng.permutation(np.arange(n_synapses) % n_neurons),
        delays=rng.uniform(min_delay, max_delay, size=n_synapses),
    )


def pmf_n_f_times(
    period: float, rate: float
) -> Tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]:
    """Compute probability mass function for number of spikes in a periodic spike train.

    For numerical stability, the PMF is computed in the log domain first.

    Args:
        period: Period of the spike train in time units.
        rate: Firing rate of the spike train in 1/time units.

    Returns:
        Tuple containing:
            - Support values (number of spikes)
            - Corresponding probabilities
    """
    ns = np.arange(period, dtype=int)
    logpns = (ns - 1) * np.log(period - ns) + ns * np.log(rate)
    logpns[1:] -= np.cumsum(np.log(ns[1:]))
    logpns -= np.max(logpns)  # to avoid overflow when exponentiating
    pns = np.exp(logpns)
    return ns, pns / np.sum(pns)


def expected_n_f_times(period: float, rate: float) -> float:
    """Compute expected number of spikes in a periodic spike train.

    Args:
        period: Period of the spike train in time units.
        rate: Firing rate of the spike train in 1/time units.

    Returns:
        Expected number of spikes in one period.
    """
    ns, pns = pmf_n_f_times(period, rate)
    return np.inner(ns, pns)


# def rand_f_times(
#     n_neurons: int,
#     period: float,
#     rate: float,
#     rng: Optional[np.random.Generator] = None,
# ) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
#     """
#     Returns a random multi-channel periodic spike train.

#     Args:
#         period (float): The cycle period of the spike train.
#         rate (float): The firing rate of the spike train.
#         n_neurons (int): The number of channels / neurons.
#         rng (np.random.Generator, optional): The random number generator. If None, uses default_rng()

#     Raises:
#         ValueError: If the period is negative.
#         ValueError: If the firing rate is negative.

#     Returns:
#         (List[npt.NDArray[np.float64]]): a multi-channel periodic spike train.
#     """
#     if period < 0.0:
#         raise ValueError(f"The period should be non-negative.")

#     if rate < 0.0:
#         raise ValueError(f"The firing rate should be non-negative.")

#     if period <= REFRACTORY_PERIOD or rate == 0.0:
#         return np.array([]), np.array([])

#     if rng is None:
#         rng = np.random.default_rng()

#     multi_f_times = []
#     multi_f_sources = []

#     ns, pns = pmf_n_f_times(period, rate)

#     for c in range(n_channels):
#         # Sample the number of spikes in [0, period)
#         n = rng.choice(ns, p=pns)
#         if n > 0:
#             multi_f_sources.append(np.full(n, c, dtype=np.intp))

#             # sample the effective poisson process in [0, period-n)
#             f_times = np.full(n, rng.uniform(0, period))
#             f_times[1:] += np.sort(rng.uniform(0, period - n, n - 1)) + np.arange(1, n)

#             # transform the effective poisson process into a periodic spike train ...
#             multi_f_times.append(f_times % period)
#         # else:
#         #     multi_f_times.append(np.array([]))

#     return np.concatenate(multi_f_sources), np.concatenate(multi_f_times)


def rand_spikes(
    n_neurons: int,
    period: float,
    rate: float,
    rng: Optional[np.random.Generator] = None,
) -> pl.DataFrame:
    """Generate random multi-channel periodic spike trains.

    Creates random spike trains for multiple neurons following a Poisson process
    within each period, ensuring proper refractory period constraints.

    Args:
        n_neurons: Number of neurons/channels.
        period: Period duration of the spike train.
        rate: Average firing rate per neuron.
        rng: Random number generator. If None, uses default_rng().

    Returns:
        DataFrame with columns: neuron, time, period.

    Raises:
        ValueError: If period or rate is negative.
    """
    if period < 0.0:
        raise ValueError(f"The period should be non-negative.")

    if rate < 0.0:
        raise ValueError(f"The firing rate should be non-negative.")

    if period <= REFRACTORY_PERIOD or rate == 0.0:
        return pl.DataFrame(
            schema={"neuron": pl.UInt32, "time": pl.Float64, "period": pl.Float64}
        )

    if rng is None:
        rng = np.random.default_rng()

    f_times = []
    f_sources = []

    ns, pns = pmf_n_f_times(period, rate)

    for l in range(n_neurons):
        # Sample the number of spikes in [0, period)
        n = rng.choice(ns, p=pns)
        if n > 0:
            f_sources.append(np.full(n, l, dtype=np.intp))

            # sample the effective poisson process in [0, period-n)
            new_f_times = np.full(n, rng.uniform(0, period))
            new_f_times[1:] += np.sort(rng.uniform(0, period - n, n - 1)) + np.arange(
                1, n
            )

            # transform the effective poisson process into a periodic spike train ...
            f_times.append(new_f_times % period)
        # else:
        #     multi_f_times.append(np.array([]))

    if len(f_times) == 0:
        return pl.DataFrame(
            schema={"neuron": pl.UInt32, "time": pl.Float64, "period": pl.Float64}
        )

    return pl.DataFrame(
        {
            "neuron": np.concatenate(f_sources),
            "time": np.concatenate(f_times),
            "period": period,
        },
        schema={"neuron": pl.UInt32, "time": pl.Float64, "period": pl.Float64},
    )


def rand_jit_f_times(
    f_times: npt.NDArray[np.float64],
    std_jitter: float,
    start: float = -np.inf,
    end: float = np.inf,
    n_iter: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> npt.NDArray[np.float64]:
    """Generate jittered version of spike train using Gibbs sampling.

    Applies Gaussian jitter to firing times while respecting refractory period
    constraints. Uses alternating sampling of even/odd indices to maintain
    temporal ordering.

    Args:
        f_times: Original firing times to be jittered.
        std_jitter: Standard deviation of Gaussian jitter noise.
        start: Lower bound of allowed time range.
        end: Upper bound of allowed time range.
        n_iter: Maximum number of Gibbs sampling iterations.
        rng: Random number generator. If None, uses default_rng().

    Returns:
        Jittered firing times maintaining refractory constraints.

    Raises:
        ValueError: If firing times are outside [start, end] range.
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

    jit_f_times = np.sort(f_times)  # make a sorted copy of the firing times

    if jit_f_times.size > 1:
        tmin = np.full(jit_f_times.size, start)
        tmax = np.full(jit_f_times.size, end)

        even = np.arange(0, jit_f_times.size, 2)
        odd = np.arange(1, jit_f_times.size, 2)

        for _ in range(n_iter):
            # fix odd indices and sample the even ones
            tmin[1:] = jit_f_times[:-1] + REFRACTORY_PERIOD
            tmax[:-1] = jit_f_times[1:] - REFRACTORY_PERIOD
            print(
                f"Sampling even indices around {jit_f_times[even]} within {tmin[even]} -- {tmax[even]}"
            )
            jit_f_times[even] = sampler(
                tmin[even],
                tmax[even],
                f_times[even],
            )

            # fix even indices and sample odd ones
            tmin[1:] = jit_f_times[:-1] + REFRACTORY_PERIOD
            tmax[:-1] = jit_f_times[1:] - REFRACTORY_PERIOD
            print(
                f"Sampling odd indices around {jit_f_times[odd]} within {tmin[odd]} -- {tmax[odd]}"
            )
            jit_f_times[odd] = sampler(
                tmin[odd],
                tmax[odd],
                f_times[odd],
            )

    elif jit_f_times.size == 1:
        jit_f_times[0] = sampler(start, end, f_times[0])

    return jit_f_times
