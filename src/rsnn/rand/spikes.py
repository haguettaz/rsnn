from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.stats import truncnorm

from rsnn import REFRACTORY_PERIOD


def pmf_n_f_times(
    period: float, rate: float
) -> Tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]:
    """Compute probability mass function for number of spikes in a periodic spike train.

    Calculates the PMF for the number of spikes that can fit within a period
    while respecting refractory period constraints. Uses log-domain computation
    for numerical stability.

    Args:
        period (float): Period of the spike train in time units.
        rate (float): Firing rate of the spike train in spikes per time unit.

    Returns:
        Tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]: A tuple containing:
            - Support values (possible number of spikes)
            - Corresponding probabilities for each spike count

    Notes:
        For numerical stability, the PMF is computed in the log domain first
        and then normalized. The computation accounts for the combinatorial
        constraints imposed by the refractory period.
    """
    ns = np.arange(period, dtype=int)
    logpns = (ns - 1) * np.log(period - ns) + ns * np.log(rate)
    logpns[1:] -= np.cumsum(np.log(ns[1:]))
    logpns -= np.max(logpns)  # to avoid overflow when exponentiating
    pns = np.exp(logpns)
    return ns, pns / np.sum(pns)


def expected_n_f_times(period: float, rate: float) -> float:
    """Compute expected number of spikes in a periodic spike train.

    Calculates the expected number of spikes that can occur within one period of a periodic spike train, taking into account the firing rate and refractory period constraints.

    Args:
        period (float): Period of the spike train in time units.
        rate (float): Firing rate of the spike train in spikes per time unit.

    Returns:
        float: Expected number of spikes in one period.

    Notes:
        Uses the probability mass function from pmf_n_f_times to compute the weighted average of possible spike counts.
    """
    ns, pns = pmf_n_f_times(period, rate)
    return np.inner(ns, pns)


def rand_spikes(
    n_neurons: int,
    periods: List[float],
    rates: List[float],
    rng: Optional[np.random.Generator] = None,
) -> pl.DataFrame:
    """Generate random multi-channel periodic spike trains.

    Creates random spike trains for multiple neurons following a Poisson process
    within each period, ensuring proper refractory period constraints. Each neuron
    generates spikes independently according to the specified rates and periods.

    Args:
        n_neurons (int): Number of neurons/channels to simulate.
        periods (list[float]): List of period durations for each spike train pattern.
        rates (list[float]): List of average firing rates per neuron (spikes per time unit).
        rng (Optional[np.random.Generator], optional): Random number generator.
            If None, uses numpy's default_rng(). Defaults to None.

    Returns:
        pl.DataFrame: Spike train data with columns 'index', 'neuron', 'time', 'period'.

    Raises:
        ValueError: If any period or rate is negative.

    Notes:
        Uses an effective Poisson process to generate spikes while respecting
        the refractory period constraint. Spikes are distributed uniformly
        within each period after accounting for refractory spacing.
    """
    if any(period < 0.0 for period in periods):
        raise ValueError(f"All periods should be non-negative.")

    if any(rate < 0.0 for rate in rates):
        raise ValueError(f"All firing rates should be non-negative.")

    if rng is None:
        rng = np.random.default_rng()

    spikes_lst = []
    for i, (period, rate) in enumerate(zip(periods, rates)):
        if period <= REFRACTORY_PERIOD or rate == 0.0:
            continue

        ns, pns = pmf_n_f_times(period, rate)

        for l in range(n_neurons):
            # Sample the number of spikes in [0, period)
            n = rng.choice(ns, p=pns)
            if n > 0:
                # sample the effective poisson process in [0, period-n)
                new_f_times = np.full(n, rng.uniform(0, period))
                new_f_times[1:] += np.sort(
                    rng.uniform(0, period - n, n - 1)
                ) + np.arange(1, n)

                spikes_lst.append(
                    pl.DataFrame(
                        [
                            pl.Series("index", [i] * n, dtype=pl.UInt32),
                            pl.Series("period", [period] * n, dtype=pl.Float64),
                            pl.Series("neuron", [l] * n, dtype=pl.UInt32),
                            pl.Series(
                                "time", np.sort(new_f_times % period), dtype=pl.Float64
                            ),
                        ],
                    )
                )

    if spikes_lst:
        return pl.concat(spikes_lst)
    return pl.DataFrame(
        schema={
            "index": pl.UInt32,
            "period": pl.Float64,
            "neuron": pl.UInt32,
            "time": pl.Float64,
        }
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
    constraints and temporal ordering. Uses alternating Gibbs sampling of
    even/odd indexed spikes to maintain feasibility.

    Args:
        f_times (npt.NDArray[np.float64]): Original firing times to be jittered.
        std_jitter (float): Standard deviation of Gaussian jitter noise.
        start (float, optional): Lower bound of allowed time range.
            Defaults to -inf.
        end (float, optional): Upper bound of allowed time range.
            Defaults to inf.
        n_iter (int, optional): Maximum number of Gibbs sampling iterations.
            Defaults to 1000.
        rng (Optional[np.random.Generator], optional): Random number generator.
            If None, uses default_rng(). Defaults to None.

    Returns:
        npt.NDArray[np.float64]: Jittered firing times maintaining refractory
            constraints and temporal ordering.

    Raises:
        ValueError: If firing times are outside the [start, end] range.

    Notes:
        Uses truncated normal sampling with alternating updates of even/odd
        indices to maintain the refractory period constraint between consecutive
        spikes while allowing controlled jitter around original times.
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

            jit_f_times[even] = sampler(
                tmin[even],
                tmax[even],
                f_times[even],
            )

            # fix even indices and sample odd ones
            tmin[1:] = jit_f_times[:-1] + REFRACTORY_PERIOD
            tmax[:-1] = jit_f_times[1:] - REFRACTORY_PERIOD

            jit_f_times[odd] = sampler(
                tmin[odd],
                tmax[odd],
                f_times[odd],
            )

    elif jit_f_times.size == 1:
        jit_f_times[0] = sampler(start, end, f_times[0])

    return jit_f_times
